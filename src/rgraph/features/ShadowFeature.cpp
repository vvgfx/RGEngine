#include "ShadowFeature.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

// defined in DeferredRenderingFeature.cpp
bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj);

namespace
{
    struct ShadowPush
    {
        glm::mat4 lightViewProj;
        glm::mat4 modelMatrix;
        VkDeviceAddress vertexBuffer;
    };
} // namespace

rgraph::ShadowFeature::ShadowFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData)
    : drawContext(drawContext), sceneData(sceneData)
{
    auto &alloc = GPUResourceAllocator::Instance();

    shadowAtlas.imageFormat = VK_FORMAT_D32_SFLOAT;
    shadowAtlas.imageExtent = {ATLAS_DIM, ATLAS_DIM, 1};

    VkImageCreateInfo imgInfo = vkinit::image_create_info(shadowAtlas.imageFormat,
                                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                          shadowAtlas.imageExtent);
    VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_GPU_ONLY,
                                      .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    alloc.create_image(&imgInfo, &allocInfo, &shadowAtlas.image, &shadowAtlas.allocation, nullptr);

    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(shadowAtlas.imageFormat, shadowAtlas.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &shadowAtlas.imageView));

    // compareEnable turns this into a sampler2DShadow: each tap is a filtered depth test.
    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_LINEAR,
                                    .minFilter = VK_FILTER_LINEAR,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .compareEnable = VK_TRUE,
                                    .compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // reverse-Z
                                    .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &shadowSampler));

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        shadowLayout = builder.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Depth-only: no fragment shader, no descriptor sets, everything through push constants.
    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(ShadowPush)};
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &depthPipeline.layout));

    VkShaderModule vert;
    if (!vkutil::load_shader_module("../shaders/shadow/shadow_depth.vert.spv", device, &vert))
    {
        fmt::println("Shadow: failed to load shadow_depth.vert.spv");
    }
    else
    {
        PipelineBuilder builder;
        builder.set_shaders(vert, VK_NULL_HANDLE);
        builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        // front-face cull halves the work and pushes acne to surfaces the camera cannot see
        builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        builder.set_multisampling_none();
        builder.set_depth_format(shadowAtlas.imageFormat);
        builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL); // reverse-Z
        builder._pipelineLayout = depthPipeline.layout;

        depthPipeline.pipeline = builder.build_pipeline(device);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, depthPipeline.pipeline, nullptr);
            vkDestroyPipelineLayout(device, depthPipeline.layout, nullptr);
            vkDestroyDescriptorSetLayout(device, shadowLayout, nullptr);
            vkDestroySampler(device, shadowSampler, nullptr);
            vkDestroyImageView(device, shadowAtlas.imageView, nullptr);
            GPUResourceAllocator::Instance().destroy_image(shadowAtlas.image, shadowAtlas.allocation);
        });

    Rendergraph::Instance().AddTrackedImage("shadowAtlas", VK_IMAGE_LAYOUT_UNDEFINED, shadowAtlas);
}

void rgraph::ShadowFeature::computeCascades()
{
    // Direction towards the sun. Falls back to a plausible angle if the scene has no directional light.
    glm::vec3 sunDir(0.4f, 0.8f, 0.3f);
    for (const GPULightingData &l : drawContext.lights)
    {
        if (l.type == 0)
        {
            sunDir = glm::normalize(glm::vec3(l.transform[2]));
            break;
        }
    }

    const float nearClip = 0.1f;
    const float farClip = settings.maxDistance;
    const float range = farClip - nearClip;

    // Practical split scheme: blends uniform and logarithmic so near cascades get the resolution.
    float splits[CASCADE_COUNT];
    for (uint32_t i = 0; i < CASCADE_COUNT; i++)
    {
        float p = float(i + 1) / float(CASCADE_COUNT);
        float log = nearClip * std::pow(farClip / nearClip, p);
        float uniform = nearClip + range * p;
        splits[i] = settings.cascadeSplitLambda * log + (1.0f - settings.cascadeSplitLambda) * uniform;
    }

    const glm::mat4 invViewProj = glm::inverse(sceneData.viewproj);
    float lastSplit = nearClip;

    for (uint32_t i = 0; i < CASCADE_COUNT; i++)
    {
        // Unproject the NDC cube, then pull the slice this cascade covers.
        glm::vec3 corners[8] = {{-1, 1, 0}, {1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {-1, 1, 1}, {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}};
        for (glm::vec3 &c : corners)
        {
            glm::vec4 inv = invViewProj * glm::vec4(c, 1.0f);
            c = glm::vec3(inv) / inv.w;
        }

        const float nearRatio = (lastSplit - nearClip) / range;
        const float farRatio = (splits[i] - nearClip) / range;
        for (uint32_t c = 0; c < 4; c++)
        {
            glm::vec3 ray = corners[c + 4] - corners[c];
            corners[c + 4] = corners[c] + ray * farRatio;
            corners[c] = corners[c] + ray * nearRatio;
        }

        glm::vec3 centre(0.0f);
        for (const glm::vec3 &c : corners)
        {
            centre += c;
        }
        centre /= 8.0f;

        // A bounding sphere keeps the cascade size constant as the camera rotates, which is what
        // stops the shadow edges from crawling.
        float radius = 0.0f;
        for (const glm::vec3 &c : corners)
        {
            radius = std::max(radius, glm::length(c - centre));
        }
        radius = std::ceil(radius * 16.0f) / 16.0f;

        glm::vec3 up = std::abs(sunDir.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
        glm::mat4 lightView = glm::lookAt(centre + sunDir * radius, centre, up);

        // reverse-Z: near and far are swapped, matching the main camera's convention
        glm::mat4 lightProj = glm::ortho(-radius, radius, -radius, radius, 3.0f * radius, -radius);

        // Snap the origin to whole texels so the cascade does not shimmer when the camera moves.
        glm::mat4 viewProj = lightProj * lightView;
        glm::vec4 origin = viewProj * glm::vec4(0, 0, 0, 1);
        origin *= float(CASCADE_RES) * 0.5f;
        glm::vec4 rounded = glm::round(origin);
        glm::vec4 offset = (rounded - origin) * (2.0f / float(CASCADE_RES));
        lightProj[3] += glm::vec4(offset.x, offset.y, 0.0f, 0.0f);

        shadowData.cascadeViewProj[i] = lightProj * lightView;
        shadowData.cascadeSplits[i] = splits[i];

        if (i == 0)
        {
            // world size of one texel in cascade 0; the shader scales it per cascade
            shadowData.params.x = (2.0f * radius) / float(CASCADE_RES);
        }

        lastSplit = splits[i];
    }

    shadowData.params.y = settings.pcfRadius;
    shadowData.params.z = float(ATLAS_DIM);
    shadowData.params.w = settings.enabled ? 1.0f : 0.0f;
}

void rgraph::ShadowFeature::Register(rgraph::Rendergraph *builder)
{
    if (depthPipeline.pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddGraphicsPass(
        "shadow-cascades",
        [](Pass &pass)
        {
            static VkClearValue clear{};
            clear.depthStencil = {0.0f, 0}; // reverse-Z clears to 0

            pass.AddDepthStencilAttachment("shadowAtlas", true, &clear);
            pass.SetRenderExtent(ATLAS_DIM, ATLAS_DIM);
            pass.CreatesBuffer("shadowData", sizeof(ShadowDataGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { renderPass(passExec); });
}

void rgraph::ShadowFeature::renderPass(PassExecution &passExec)
{
    computeCascades();

    AllocatedBuffer buffer = passExec.allocatedBuffers["shadowData"];
    *(ShadowDataGPU *)buffer.info.pMappedData = shadowData;

    frameSet = passExec.frameDescriptor->allocate(passExec._device, shadowLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, buffer.buffer, sizeof(ShadowDataGPU), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, shadowAtlas.imageView, shadowSampler, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(passExec._device, frameSet);

    if (!settings.enabled)
    {
        return;
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline.pipeline);

    uint32_t draws = 0;

    for (uint32_t c = 0; c < CASCADE_COUNT; c++)
    {
        // Each cascade owns one quadrant of the atlas; scissor keeps its clear and draws contained.
        const float offsetX = float(c & 1) * float(CASCADE_RES);
        const float offsetY = float(c >> 1) * float(CASCADE_RES);

        VkViewport viewport{offsetX, offsetY, float(CASCADE_RES), float(CASCADE_RES), 0.0f, 1.0f};
        vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

        VkRect2D scissor{{int32_t(offsetX), int32_t(offsetY)}, {CASCADE_RES, CASCADE_RES}};
        vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

        for (const RenderObject &obj : drawContext.OpaqueSurfaces)
        {
            // Without this every cascade redraws the whole scene: 4 x 2906 draws. Each cascade only
            // covers a slice of the view frustum, so most objects fall outside it.
            if (!is_visible(obj, shadowData.cascadeViewProj[c]))
            {
                continue;
            }

            ShadowPush push{shadowData.cascadeViewProj[c], obj.modelMatrix, obj.vertexBufferAddress};
            vkCmdPushConstants(passExec.cmd, depthPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &push);

            vkCmdBindIndexBuffer(passExec.cmd, obj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(passExec.cmd, obj.indexCount, 1, obj.firstIndex, 0, 0);
            draws++;
        }
    }

    passExec.drawCalls = float(draws);
}
