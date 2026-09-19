#include "LightCullFeature.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <cmath>

rgraph::LightCullFeature::LightCullFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData,
                                           VkExtent3D screenExtent)
    : drawContext(drawContext), sceneData(sceneData), screenExtent(screenExtent)
{
    tileCount = {(screenExtent.width + TILE_SIZE - 1) / TILE_SIZE, (screenExtent.height + TILE_SIZE - 1) / TILE_SIZE};
    const uint32_t tiles = tileCount.x * tileCount.y;

    auto &alloc = GPUResourceAllocator::Instance();

    gridBuffer = alloc.create_buffer(sizeof(uint32_t) * tiles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    indexBuffer = alloc.create_buffer(sizeof(uint32_t) * tiles * MAX_LIGHTS_PER_TILE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    fmt::println("LightCull: {}x{} tiles ({} total), index buffer {:.1f} MB", tileCount.x, tileCount.y, tiles,
                 sizeof(uint32_t) * tiles * MAX_LIGHTS_PER_TILE / 1048576.0);

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // position
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // grid
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // indices
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // lights
        builder.add_binding(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // scene
        cullLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // grid
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // indices
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // dims
        readLayout = builder.build(device, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
    }

    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_NEAREST,
                                    .minFilter = VK_FILTER_NEAREST,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));

    // The grid dimensions never change, so this small uniform is written once at init.
    infoBuffer = alloc.create_buffer(sizeof(glm::uvec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    *(glm::uvec4 *)infoBuffer.info.pMappedData = glm::uvec4(tileCount.x, tileCount.y, 0, 0);

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3}};
    descriptorAllocator.init(device, 10, sizes);

    // The read set only references persistent buffers, so it is written once.
    readSet = descriptorAllocator.allocate(device, readLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, gridBuffer.buffer, sizeof(uint32_t) * tiles, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, indexBuffer.buffer, sizeof(uint32_t) * tiles * MAX_LIGHTS_PER_TILE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, infoBuffer.buffer, sizeof(glm::uvec4), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(device, readSet);
    }

    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .setLayoutCount = 1,
                                          .pSetLayouts = &cullLayout,
                                          .pushConstantRangeCount = 1,
                                          .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &cullPipelineLayout));

    VkShaderModule module;
    if (!vkutil::load_shader_module("../shaders/lighting/light_cull.comp.spv", device, &module))
    {
        fmt::println("LightCull: failed to load light_cull.comp.spv");
    }
    else
    {
        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"};
        VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = cullPipelineLayout};
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &cullPipeline));
        vkDestroyShaderModule(device, module, nullptr);
    }

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, cullPipeline, nullptr);
            vkDestroyPipelineLayout(device, cullPipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, cullLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, readLayout, nullptr);
            vkDestroySampler(device, sampler, nullptr);
            descriptorAllocator.destroy_pools(device);

            auto &a = GPUResourceAllocator::Instance();
            a.destroy_buffer(gridBuffer);
            a.destroy_buffer(indexBuffer);
            a.destroy_buffer(infoBuffer);
        });
}

void rgraph::LightCullFeature::RegisterCullPass(rgraph::Rendergraph *builder)
{
    if (cullPipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddComputePass(
        "light-cull",
        [](Pass &pass)
        {
            pass.ReadsImage("position_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.CreatesBuffer("cullLights", sizeof(LightBlockGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            pass.CreatesBuffer("cullScene", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { run(passExec); });
}

void rgraph::LightCullFeature::run(PassExecution &passExec)
{
    AllocatedBuffer lightBuffer = passExec.allocatedBuffers["cullLights"];
    AllocatedBuffer sceneBuffer = passExec.allocatedBuffers["cullScene"];

    auto *lights = (LightBlockGPU *)lightBuffer.info.pMappedData;
    lights->numLights = std::min<int>(int(drawContext.lights.size()), int(MAX_LIGHTS));
    for (int i = 0; i < lights->numLights; i++)
    {
        const GPULightingData &src = drawContext.lights[i];
        lights->lights[i] = LightGPU{src.transform,
                                     src.color,
                                     src.intensity * (src.type == 0 ? drawContext.sunIntensityScale : drawContext.localIntensityScale),
                                     src.range,
                                     src.type,
                                     -1,
                                     {}};
    }

    *(GPUSceneData *)sceneBuffer.info.pMappedData = sceneData;

    VkDescriptorSet set = passExec.frameDescriptor->allocate(passExec._device, cullLayout);
    DescriptorWriter writer;
    writer.write_image(0, passExec.allocatedImages["position_gbuf"].imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_buffer(1, gridBuffer.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    writer.write_buffer(2, indexBuffer.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    writer.write_buffer(3, lightBuffer.buffer, sizeof(LightBlockGPU), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_buffer(4, sceneBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(passExec._device, set);

    PushConstants push{glm::uvec4(tileCount.x, tileCount.y, passExec._drawExtent.width, passExec._drawExtent.height)};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    vkCmdDispatch(passExec.cmd, tileCount.x, tileCount.y, 1);
    passExec.dispatchCalls = float(tileCount.x * tileCount.y);

    // The rendergraph emits no buffer barriers, and this must be issued here rather than in the
    // composite: that is a graphics pass, and inside vkCmdBeginRendering a COMPUTE -> FRAGMENT
    // barrier is illegal.
    VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT};
    VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(passExec.cmd, &dep);
}
