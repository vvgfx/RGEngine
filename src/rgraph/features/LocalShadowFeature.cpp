#include "LocalShadowFeature.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    struct ShadowPush
    {
        glm::mat4 lightViewProj;
        glm::mat4 modelMatrix;
        VkDeviceAddress vertexBuffer;
    };

    /// Cube face basis. Order must match localShadowFace() in local_shadow.glsl.
    struct FaceDir
    {
        glm::vec3 forward;
        glm::vec3 up;
    };

    constexpr FaceDir FACES[6] = {
        {{1, 0, 0}, {0, -1, 0}}, {{-1, 0, 0}, {0, -1, 0}}, {{0, 1, 0}, {0, 0, 1}},
        {{0, -1, 0}, {0, 0, -1}}, {{0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, -1, 0}},
    };
} // namespace

rgraph::LocalShadowFeature::LocalShadowFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData)
    : drawContext(drawContext), sceneData(sceneData)
{
    auto &alloc = GPUResourceAllocator::Instance();

    atlas.imageFormat = VK_FORMAT_D32_SFLOAT;
    atlas.imageExtent = {ATLAS_DIM, ATLAS_DIM, 1};

    VkImageCreateInfo imgInfo =
        vkinit::image_create_info(atlas.imageFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, atlas.imageExtent);
    VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_GPU_ONLY,
                                      .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    alloc.create_image(&imgInfo, &allocInfo, &atlas.image, &atlas.allocation, nullptr);

    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(atlas.imageFormat, atlas.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &atlas.imageView));

    // compareEnable makes each PCF tap a filtered depth test rather than a raw fetch
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
        setLayout = builder.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Depth-only, so it reuses the cascade shader: no fragment stage, no descriptor sets.
    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(ShadowPush)};
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &depthPipeline.layout));

    VkShaderModule vert;
    if (!vkutil::load_shader_module("../shaders/shadow/shadow_depth.vert.spv", device, &vert))
    {
        fmt::println("LocalShadow: failed to load shadow_depth.vert.spv");
    }
    else
    {
        PipelineBuilder builder;
        builder.set_shaders(vert, VK_NULL_HANDLE);
        builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        // front-face cull pushes acne onto surfaces the camera cannot see
        builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        builder.set_multisampling_none();
        builder.set_depth_format(atlas.imageFormat);
        builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL); // reverse-Z
        builder._pipelineLayout = depthPipeline.layout;

        depthPipeline.pipeline = builder.build_pipeline(device);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    shadowIndexByLight.assign(128, -1);

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, depthPipeline.pipeline, nullptr);
            vkDestroyPipelineLayout(device, depthPipeline.layout, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkDestroySampler(device, shadowSampler, nullptr);
            vkDestroyImageView(device, atlas.imageView, nullptr);
            GPUResourceAllocator::Instance().destroy_image(atlas.image, atlas.allocation);
        });

    Rendergraph::Instance().AddTrackedImage("localShadowAtlas", VK_IMAGE_LAYOUT_UNDEFINED, atlas);
}

int rgraph::LocalShadowFeature::ShadowIndexFor(int lightIndex) const
{
    return lightIndex >= 0 && lightIndex < int(shadowIndexByLight.size()) ? shadowIndexByLight[lightIndex] : -1;
}

void rgraph::LocalShadowFeature::selectLights()
{
    casters.clear();
    std::fill(shadowIndexByLight.begin(), shadowIndexByLight.end(), -1);

    if (!settings.enabled)
    {
        return;
    }

    const glm::vec3 camera = glm::vec3(sceneData.cameraPos);

    for (int i = 0; i < int(drawContext.lights.size()); i++)
    {
        const GPULightingData &light = drawContext.lights[i];
        if (light.type == 0) // directional lights are handled by the cascades
        {
            continue;
        }

        const glm::vec3 pos = glm::vec3(light.transform[3]);
        const float distSq = glm::dot(pos - camera, pos - camera);

        // Prefer bright, nearby lights. Lights whose reach cannot even touch the camera's vicinity
        // contribute little on screen, so falling off with distance is the right bias.
        casters.push_back({i, light.intensity * light.range / (distSq + 1.0f)});
    }

    const int budget = std::min<int>(settings.maxLights, int(MAX_LIGHTS));
    if (int(casters.size()) > budget)
    {
        std::partial_sort(casters.begin(), casters.begin() + budget, casters.end(),
                          [](const Caster &a, const Caster &b) { return a.importance > b.importance; });
        casters.resize(budget);
    }

    for (int slot = 0; slot < int(casters.size()); slot++)
    {
        if (casters[slot].lightIndex < int(shadowIndexByLight.size()))
        {
            shadowIndexByLight[casters[slot].lightIndex] = slot;
        }
    }
}

void rgraph::LocalShadowFeature::Register(rgraph::Rendergraph *builder)
{
    if (depthPipeline.pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddGraphicsPass(
        "local-shadows",
        [](Pass &pass)
        {
            static VkClearValue clear{};
            clear.depthStencil = {0.0f, 0}; // reverse-Z clears to 0

            pass.AddDepthStencilAttachment("localShadowAtlas", true, &clear);
            pass.SetRenderExtent(ATLAS_DIM, ATLAS_DIM);
            pass.CreatesBuffer("localShadowData", sizeof(ShadowDataGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { renderPass(passExec); });
}

void rgraph::LocalShadowFeature::renderPass(PassExecution &passExec)
{
    selectLights();

    shadowData.params = glm::vec4(settings.pcfRadius, float(ATLAS_DIM), settings.normalBias, settings.enabled ? 1.0f : 0.0f);

    // Build every face matrix and tile rect first, so the descriptor is complete even for slots
    // that end up with no casters.
    for (uint32_t slot = 0; slot < MAX_LIGHTS; slot++)
    {
        for (uint32_t face = 0; face < 6; face++)
        {
            const uint32_t index = slot * 6 + face;
            const uint32_t tile = index;

            const float originX = float(tile % TILES_PER_ROW) * float(TILE_RES);
            const float originY = float(tile / TILES_PER_ROW) * float(TILE_RES);

            shadowData.tileRect[index] =
                glm::vec4(originX / float(ATLAS_DIM), originY / float(ATLAS_DIM), float(TILE_RES) / float(ATLAS_DIM), float(TILE_RES) / float(ATLAS_DIM));

            if (slot < casters.size())
            {
                const GPULightingData &light = drawContext.lights[casters[slot].lightIndex];
                const glm::vec3 pos = glm::vec3(light.transform[3]);
                const float range = std::max(light.range, 0.1f);

                const glm::mat4 view = glm::lookAt(pos, pos + FACES[face].forward, FACES[face].up);
                // reverse-Z: near and far swapped, matching the main camera
                const glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, range, 0.05f);

                shadowData.faceViewProj[index] = proj * view;
            }
            else
            {
                shadowData.faceViewProj[index] = glm::mat4(1.0f);
            }
        }
    }

    AllocatedBuffer buffer = passExec.allocatedBuffers["localShadowData"];
    *(ShadowDataGPU *)buffer.info.pMappedData = shadowData;

    frameSet = passExec.frameDescriptor->allocate(passExec._device, setLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, buffer.buffer, sizeof(ShadowDataGPU), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, atlas.imageView, shadowSampler, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(passExec._device, frameSet);

    if (casters.empty())
    {
        return;
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline.pipeline);

    uint32_t draws = 0;
    uint32_t tris = 0;
    std::vector<const RenderObject *> nearby;

    for (uint32_t slot = 0; slot < casters.size(); slot++)
    {
        const GPULightingData &light = drawContext.lights[casters[slot].lightIndex];
        const glm::vec3 lightPos = glm::vec3(light.transform[3]);
        const float range = std::max(light.range, 0.1f);

        // Cull once per light rather than once per face: the surviving set is small enough that
        // redrawing it for all six faces is cheaper than six frustum passes over the whole scene.
        nearby.clear();
        for (const RenderObject &obj : drawContext.OpaqueSurfaces)
        {
            const glm::vec3 centre = glm::vec3(obj.modelMatrix * glm::vec4(obj.bounds.origin, 1.0f));
            const glm::vec3 scale{glm::length(glm::vec3(obj.modelMatrix[0])), glm::length(glm::vec3(obj.modelMatrix[1])),
                                  glm::length(glm::vec3(obj.modelMatrix[2]))};
            const float radius = obj.bounds.sphereRadius * std::max(scale.x, std::max(scale.y, scale.z));

            const glm::vec3 delta = centre - lightPos;
            const float reach = range + radius;
            if (glm::dot(delta, delta) <= reach * reach)
            {
                nearby.push_back(&obj);
            }
        }

        if (nearby.empty())
        {
            continue;
        }

        for (uint32_t face = 0; face < 6; face++)
        {
            const uint32_t index = slot * 6 + face;
            const float originX = float(index % TILES_PER_ROW) * float(TILE_RES);
            const float originY = float(index / TILES_PER_ROW) * float(TILE_RES);

            VkViewport viewport{originX, originY, float(TILE_RES), float(TILE_RES), 0.0f, 1.0f};
            vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

            VkRect2D scissor{{int32_t(originX), int32_t(originY)}, {TILE_RES, TILE_RES}};
            vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

            for (const RenderObject *obj : nearby)
            {
                ShadowPush push{shadowData.faceViewProj[index], obj->modelMatrix, obj->vertexBufferAddress};
                vkCmdPushConstants(passExec.cmd, depthPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &push);

                vkCmdBindIndexBuffer(passExec.cmd, obj->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(passExec.cmd, obj->indexCount, 1, obj->firstIndex, 0, 0);
                draws++;
                tris += obj->indexCount / 3;
            }
        }
    }

    passExec.drawCalls = float(draws);
    passExec.triangles = float(tris);
}
