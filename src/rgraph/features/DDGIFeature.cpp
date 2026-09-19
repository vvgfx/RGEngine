#include "DDGIFeature.h"
#include "BindlessTextures.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include "vk_images.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <cfloat>
#include <glm/gtc/matrix_transform.hpp>
#include <random>

namespace
{
    constexpr int IRR_TILE = 8;   // 6 interior + 1-texel border on each side
    constexpr int DIST_TILE = 16; // 14 interior + border

    VkPipeline createComputePipeline(VkDevice device, const char *path, VkPipelineLayout layout)
    {
        VkShaderModule module;
        if (!vkutil::load_shader_module(path, device, &module))
        {
            fmt::println("DDGI: failed to load {}", path);
            return VK_NULL_HANDLE;
        }

        VkPipelineShaderStageCreateInfo stage{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .module = module,
                                              .pName = "main"};

        VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = layout};

        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        vkDestroyShaderModule(device, module, nullptr);
        return pipeline;
    }
} // namespace

rgraph::DDGIFeature::DDGIFeature(VulkanEngine *engine, VkDevice device, const AccelStructure &accel, const DrawContext &sceneSnapshot,
                                 DrawContext &drawContext, GPUSceneData &sceneData, VkDescriptorSetLayout sceneDataLayout, VkFormat colorFormat,
                                 VkFormat depthFormat, DeletionQueue &delQueue)
    : engine(engine), accel(accel), drawContext(drawContext), sceneData(sceneData), colorFormat(colorFormat), depthFormat(depthFormat),
      sceneDataLayout(sceneDataLayout)
{
    fitVolumeToScene(sceneSnapshot);
    createImages(device, delQueue);
    createPipelines(device, delQueue);
}

void rgraph::DDGIFeature::fitVolumeToScene(const DrawContext &snapshot)
{
    glm::vec3 lo(FLT_MAX);
    glm::vec3 hi(-FLT_MAX);

    for (const RenderObject &obj : snapshot.OpaqueSurfaces)
    {
        glm::vec3 centre = glm::vec3(obj.modelMatrix * glm::vec4(obj.bounds.origin, 1.0f));

        // conservative world extents: abs of the rotation/scale basis applied to the local extents
        glm::mat3 basis = glm::mat3(obj.modelMatrix);
        glm::vec3 extents = glm::abs(basis[0]) * obj.bounds.extents.x + glm::abs(basis[1]) * obj.bounds.extents.y +
                            glm::abs(basis[2]) * obj.bounds.extents.z;

        lo = glm::min(lo, centre - extents);
        hi = glm::max(hi, centre + extents);
    }

    if (lo.x > hi.x)
    {
        lo = glm::vec3(-10.0f);
        hi = glm::vec3(10.0f);
    }

    // Inset slightly so the outermost probes sit inside the scene rather than out in empty space.
    glm::vec3 size = hi - lo;
    lo -= size * 0.02f;
    hi += size * 0.02f;

    volumeOrigin = lo;
    volumeSpacing = (hi - lo) / glm::vec3(glm::max(probeCounts - 1, glm::ivec3(1)));
    maxRayDistance = glm::length(hi - lo);

    fmt::println("DDGI: volume [{:.1f} {:.1f} {:.1f}] -> [{:.1f} {:.1f} {:.1f}], spacing {:.2f} {:.2f} {:.2f}, {} probes", lo.x, lo.y, lo.z, hi.x,
                 hi.y, hi.z, volumeSpacing.x, volumeSpacing.y, volumeSpacing.z, numProbes());
}

void rgraph::DDGIFeature::createImages(VkDevice device, DeletionQueue &delQueue)
{
    auto &alloc = GPUResourceAllocator::Instance();

    // probeRayData is rays x probes, so the probe count is bounded by the 2D image limit.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(engine->GetPhysicalDevice(), &props);
    while (numProbes() > int(props.limits.maxImageDimension2D) && probeCounts.y > 1)
    {
        probeCounts.y--;
        fmt::println("DDGI: probe count exceeded maxImageDimension2D, reducing Y to {}", probeCounts.y);
    }

    const int tilesPerRow = probeCounts.x * probeCounts.y;
    const int tileRows = probeCounts.z;

    const VkImageUsageFlags storageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageUsageFlags sampledUsage = storageUsage | VK_IMAGE_USAGE_SAMPLED_BIT;

    probeRayData = alloc.create_image({uint32_t(settings.raysPerProbe), uint32_t(numProbes()), 1}, VK_FORMAT_R16G16B16A16_SFLOAT, storageUsage);
    probeIrradiance =
        alloc.create_image({uint32_t(tilesPerRow * IRR_TILE), uint32_t(tileRows * IRR_TILE), 1}, VK_FORMAT_R16G16B16A16_SFLOAT, sampledUsage);
    probeDistance = alloc.create_image({uint32_t(tilesPerRow * DIST_TILE), uint32_t(tileRows * DIST_TILE), 1}, VK_FORMAT_R16G16_SFLOAT, sampledUsage);

    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_LINEAR,
                                    .minFilter = VK_FILTER_LINEAR,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &probeSampler));

    // One-time move to GENERAL, plus a clear. The clear is not optional: uninitialised fp16 reads as
    // NaN and the hysteresis blend would make a single NaN texel permanent.
    engine->immediate_submit(
        [&](VkCommandBuffer cmd)
        {
            VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};

            VkClearColorValue zero{};
            VkClearColorValue far{};
            far.float32[0] = maxRayDistance;
            far.float32[1] = maxRayDistance * maxRayDistance;

            for (AllocatedImage *img : {&probeRayData, &probeIrradiance, &probeDistance})
            {
                vkutil::transition_image(cmd, img->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
                vkCmdClearColorImage(cmd, img->image, VK_IMAGE_LAYOUT_GENERAL, img == &probeDistance ? &far : &zero, 1, &range);
            }
        });

    delQueue.push_function(
        [device, this]()
        {
            auto &a = GPUResourceAllocator::Instance();
            vkDestroySampler(device, probeSampler, nullptr);
            a.destroy_image(probeRayData);
            a.destroy_image(probeIrradiance);
            a.destroy_image(probeDistance);
        });
}

void rgraph::DDGIFeature::createPipelines(VkDevice device, DeletionQueue &delQueue)
{
    // Shared by every DDGI shader; bound at set 0 in compute and set 3 in the composite pass.
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // volume
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // probe ray data
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // irradiance (write)
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // distance (write)
        builder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // irradiance (sample)
        builder.add_binding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // distance (sample)
        builder.add_binding(6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // lights
        builder.add_binding(7, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR); // TLAS, for composite shadow rays

        VkDescriptorBindingFlags bindingFlags[8] = {};
        bindingFlags[7] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                              .bindingCount = 8,
                                                              .pBindingFlags = bindingFlags};

        ddgiLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, &flagsInfo);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        sceneLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};

    VkDescriptorSetLayout traceSets[] = {ddgiLayout, sceneLayout, BindlessTextures::Instance().GetLayout()};
    VkPipelineLayoutCreateInfo traceInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = 3,
                                         .pSetLayouts = traceSets,
                                         .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &traceInfo, nullptr, &tracePipelineLayout));

    VkPipelineLayoutCreateInfo blendInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = 1,
                                         .pSetLayouts = &ddgiLayout,
                                         .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &blendInfo, nullptr, &blendPipelineLayout));

    if (engine->IsRayQuerySupported())
    {
        tracePipeline = createComputePipeline(device, "../shaders/ddgi/ddgi_trace_rq.comp.spv", tracePipelineLayout);
    }
    blendIrradiancePipeline = createComputePipeline(device, "../shaders/ddgi/ddgi_blend_irradiance.comp.spv", blendPipelineLayout);
    blendDistancePipeline = createComputePipeline(device, "../shaders/ddgi/ddgi_blend_distance.comp.spv", blendPipelineLayout);

    // --- probe overlay ---
    {
        VkPushConstantRange debugRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(glm::vec4)};
        VkDescriptorSetLayout debugSets[] = {ddgiLayout, sceneDataLayout};
        VkPipelineLayoutCreateInfo debugInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                             .setLayoutCount = 2,
                                             .pSetLayouts = debugSets,
                                             .pushConstantRangeCount = 1,
                                             .pPushConstantRanges = &debugRange};
        VK_CHECK(vkCreatePipelineLayout(device, &debugInfo, nullptr, &debugPipelineLayout));

        VkShaderModule vert, frag;
        if (vkutil::load_shader_module("../shaders/ddgi/debug_probes.vert.spv", device, &vert) &&
            vkutil::load_shader_module("../shaders/ddgi/debug_probes.frag.spv", device, &frag))
        {
            PipelineBuilder builder;
            builder.set_shaders(vert, frag);
            builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
            builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
            builder.set_multisampling_none();
            builder.disable_blending();
            builder.set_color_attachment_format(colorFormat);
            builder.set_depth_format(depthFormat);
            // reverse-Z, so nearer fragments compare GREATER
            builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
            builder._pipelineLayout = debugPipelineLayout;

            debugPipeline = builder.build_pipeline(device);

            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
        }
    }

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, tracePipeline, nullptr);
            vkDestroyPipeline(device, blendIrradiancePipeline, nullptr);
            vkDestroyPipeline(device, blendDistancePipeline, nullptr);
            vkDestroyPipeline(device, debugPipeline, nullptr);
            vkDestroyPipelineLayout(device, debugPipelineLayout, nullptr);
            vkDestroyPipelineLayout(device, tracePipelineLayout, nullptr);
            vkDestroyPipelineLayout(device, blendPipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, ddgiLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, sceneLayout, nullptr);
        });
}

void rgraph::DDGIFeature::memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst)
{
    VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .srcStageMask = src,
                             .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                             .dstStageMask = dst,
                             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT};

    VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(cmd, &dep);
}

bool rgraph::DDGIFeature::active() const
{
    return settings.enabled && tracePipeline != VK_NULL_HANDLE && accel.IsValid();
}

void rgraph::DDGIFeature::Register(rgraph::Rendergraph *builder)
{
    // This pass is always declared: the composite pass binds frameSet unconditionally, so it must be
    // allocated and written every frame even when DDGI is switched off. Only the work is gated.
    builder->AddComputePass(
        "ddgi-trace",
        [](Pass &pass)
        {
            // Nothing is declared about the probe atlases: they are untracked, and naming an
            // untracked image here would throw inside Rendergraph::Run.
            pass.CreatesBuffer("ddgiVolume", sizeof(VolumeGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            pass.CreatesBuffer("ddgiLights", sizeof(LightBlockGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { tracePass(passExec); });

    // The graph rebuilds every frame, so simply not declaring these is a complete disable path.
    if (!active())
    {
        return;
    }

    builder->AddComputePass(
        "ddgi-blend-irradiance", [](Pass &pass) {}, [&](PassExecution &passExec) { blendPass(passExec, blendIrradiancePipeline, IRR_TILE); });

    builder->AddComputePass(
        "ddgi-blend-distance", [](Pass &pass) {}, [&](PassExecution &passExec) { blendPass(passExec, blendDistancePipeline, DIST_TILE); });
}

void rgraph::DDGIFeature::tracePass(PassExecution &passExec)
{
    AllocatedBuffer volumeBuffer = passExec.allocatedBuffers["ddgiVolume"];
    AllocatedBuffer lightBuffer = passExec.allocatedBuffers["ddgiLights"];

    auto *volume = (VolumeGPU *)volumeBuffer.info.pMappedData;
    volume->origin = glm::vec4(volumeOrigin, 0.0f);
    volume->spacing = glm::vec4(volumeSpacing, 0.0f);
    volume->counts = glm::ivec4(probeCounts, settings.raysPerProbe);
    volume->blend = glm::vec4(settings.hysteresis, settings.normalBias * volumeSpacing.x, settings.viewBias * volumeSpacing.x, maxRayDistance);
    volume->misc = glm::vec4(settings.depthSharpness, settings.irradianceGamma, float(frameIndex), active() ? 1.0f : 0.0f);
    volume->skyColor = glm::vec4(settings.skyColor, 1.0f);
    volume->flags = glm::vec4(settings.sunShadows && accel.IsValid() ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);

    auto *lights = (LightBlockGPU *)lightBuffer.info.pMappedData;
    lights->numLights = std::min<int>(int(drawContext.lights.size()), MAX_LIGHTS);
    for (int i = 0; i < lights->numLights; i++)
    {
        const GPULightingData &src = drawContext.lights[i];
        lights->lights[i] =
            LightGPU{src.transform, src.color, src.intensity * drawContext.lightIntensityScale, src.range, src.type, {}};
    }

    // A fresh random rotation each frame is what lets 128 rays converge through hysteresis.
    static std::mt19937 rng{1337};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    glm::vec3 axis = glm::normalize(glm::vec3(dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f) + glm::vec3(1e-4f));
    pushConstants.rayRotation = glm::rotate(glm::mat4(1.0f), dist(rng) * 6.2831853f, axis);
    pushConstants.flags = glm::vec4(settings.shadowRays ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);

    frameSet = passExec.frameDescriptor->allocate(passExec._device, ddgiLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, volumeBuffer.buffer, sizeof(VolumeGPU), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, probeRayData.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.write_image(2, probeIrradiance.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.write_image(3, probeDistance.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.write_image(4, probeIrradiance.imageView, probeSampler, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(5, probeDistance.imageView, probeSampler, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_buffer(6, lightBuffer.buffer, sizeof(LightBlockGPU), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    VkAccelerationStructureKHR tlasHandle = accel.GetTLAS();
    if (accel.IsValid())
    {
        writer.write_accel(7, &tlasHandle);
    }

    writer.update_set(passExec._device, frameSet);

    VkDescriptorSet sceneSet = passExec.frameDescriptor->allocate(passExec._device, sceneLayout);

    DescriptorWriter sceneWriter;
    sceneWriter.write_accel(0, &tlasHandle);
    sceneWriter.write_buffer(1, accel.GetGeometryBuffer(), accel.GetGeometryBufferSize(), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    sceneWriter.update_set(passExec._device, sceneSet);

    if (!active())
    {
        return;
    }

    // Frames N-1/N-2 may still be sampling the atlases in their composite pass: draw() only waits on
    // the current frame slot's fence, so this is a real cross-submission WAR hazard.
    memoryBarrier(passExec.cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    VkDescriptorSet sets[] = {frameSet, sceneSet, BindlessTextures::Instance().GetSet()};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tracePipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tracePipelineLayout, 0, 3, sets, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, tracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    const uint32_t groupsX = (settings.raysPerProbe + 63) / 64;
    vkCmdDispatch(passExec.cmd, groupsX, uint32_t(numProbes()), 1);

    passExec.dispatchCalls = float(groupsX * numProbes());

    memoryBarrier(passExec.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    frameIndex++;
}

void rgraph::DDGIFeature::blendPass(PassExecution &passExec, VkPipeline pipeline, int tileSize)
{
    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blendPipelineLayout, 0, 1, &frameSet, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, blendPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    vkCmdDispatch(passExec.cmd, uint32_t(numProbes()), 1, 1);
    passExec.dispatchCalls = float(numProbes());

    // The distance blend is the last DDGI pass, so its barrier is the one the composite pass relies
    // on. It cannot live in the composite lambda: that runs inside vkCmdBeginRendering, where a
    // COMPUTE -> FRAGMENT barrier is illegal.
    memoryBarrier(passExec.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
}

void rgraph::DDGIFeature::RegisterDebug(rgraph::Rendergraph *builder)
{
    if (!settings.showProbes || debugPipeline == VK_NULL_HANDLE || frameSet == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddGraphicsPass(
        "ddgi-debug-probes",
        [](Pass &pass)
        {
            pass.AddColorAttachment("drawImage", true, nullptr);
            // every graphics pass must name a depth attachment, or Run feeds an uninitialised view
            // to vkCmdBeginRendering
            pass.AddDepthStencilAttachment("depth_gbuf", true, nullptr);
            pass.CreatesBuffer("ddgiDebugScene", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { debugProbePass(passExec); });
}

void rgraph::DDGIFeature::debugProbePass(PassExecution &passExec)
{
    AllocatedBuffer sceneBuffer = passExec.allocatedBuffers["ddgiDebugScene"];
    *(GPUSceneData *)sceneBuffer.info.pMappedData = sceneData;

    VkDescriptorSet sceneSet = passExec.frameDescriptor->allocate(passExec._device, sceneDataLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, sceneBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(passExec._device, sceneSet);

    VkDescriptorSet sets[] = {frameSet, sceneSet};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, debugPipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, debugPipelineLayout, 0, 2, sets, 0, nullptr);

    const float minSpacing = glm::min(glm::min(volumeSpacing.x, volumeSpacing.y), volumeSpacing.z);
    glm::vec4 params(minSpacing * settings.probeRadius, 0.0f, 0.0f, 0.0f);
    vkCmdPushConstants(passExec.cmd, debugPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec4), &params);

    VkViewport viewport{0.0f, 0.0f, float(passExec._drawExtent.width), float(passExec._drawExtent.height), 0.0f, 1.0f};
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, {passExec._drawExtent.width, passExec._drawExtent.height}};
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    const uint32_t vertsPerProbe = 8 * 8 * 6; // SLICES * STACKS * 6, matching debug_probes.vert
    vkCmdDraw(passExec.cmd, vertsPerProbe, uint32_t(numProbes()), 0, 0);

    passExec.drawCalls = 1;
    passExec.triangles = float(vertsPerProbe / 3 * numProbes());
}
