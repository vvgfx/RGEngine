#include "rgraph/features/DeferredRenderingFeature.h"
#include "DeferredRenderingFeature.h"
#include "GPUResourceAllocator.h"
#include "rgraph/Rendergraph.h"
#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <vulkan/vulkan_core.h>

bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj);

rgraph::DeferredRenderingFeature::DeferredRenderingFeature(DrawContext &drawContext, VkDevice _device, GPUSceneData &gpuSceneData,
                                                           VkDescriptorSetLayout gpuSceneLayout, MaterialSystemCreateInfo &materialSystemCreateInfo,
                                                           DeletionQueue &delQueue)
    : drawContext(drawContext), gpuSceneData(gpuSceneData)
{
    _gpuSceneDataDescriptorLayout = gpuSceneLayout;

    // descriptor set for lights.
    {
        DescriptorLayoutBuilder layoutBuilder;
        layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        lightDescriptorSetLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // default sampler - nearest
    {
        VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

        sampl.magFilter = VK_FILTER_NEAREST;
        sampl.minFilter = VK_FILTER_NEAREST;

        vkCreateSampler(_device, &sampl, nullptr, &defaultSampler);
    }

    // descriptor set for composite pass inputs.
    {
        DescriptorLayoutBuilder layoutBuilder;
        layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // position
        layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // normal
        layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // albedo
        layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // metallic-roughness
        layoutBuilder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // shadow map
        compDescriptorSetLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // shadow map sampler - linear, clamp so lookups outside the light frustum read the edge
    {
        VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampl.magFilter = VK_FILTER_LINEAR;
        sampl.minFilter = VK_FILTER_LINEAR;
        sampl.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampl.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampl.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(_device, &sampl, nullptr, &shadowSampler);
    }

    // create images here and add them to be tracked.
    createImages(delQueue);

    createPipelines(materialSystemCreateInfo);

    delQueue.push_function(
        [_device, this]()
        {
            vkDestroySampler(_device, defaultSampler, nullptr);
            vkDestroySampler(_device, shadowSampler, nullptr);
            vkDestroyDescriptorSetLayout(_device, lightDescriptorSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, compDescriptorSetLayout, nullptr);
            vkDestroyPipelineLayout(_device, geometryPipeline.layout, nullptr);
            vkDestroyPipelineLayout(_device, compositePipeline.layout, nullptr);
            vkDestroyPipelineLayout(_device, transparentPipeline.layout, nullptr);
            vkDestroyPipelineLayout(_device, shadowPipeline.layout, nullptr);
            vkDestroyPipeline(_device, geometryPipeline.pipeline, nullptr);
            vkDestroyPipeline(_device, compositePipeline.pipeline, nullptr);
            vkDestroyPipeline(_device, transparentPipeline.pipeline, nullptr);
            vkDestroyPipeline(_device, shadowPipeline.pipeline, nullptr);
        });
}

void rgraph::DeferredRenderingFeature::Register(rgraph::Rendergraph *builder)
{

    // Shadow pass: scene depth from the sun's view; runs first so the composite can sample it.
    builder->AddGraphicsPass(
        "Shadow Pass",
        [](Pass &pass)
        {
            static VkClearValue depthClear{};
            depthClear.depthStencil = {1.0f, 0};
            pass.AddDepthStencilAttachment("shadowMap", true, &depthClear);
            pass.CreatesBuffer("shadowSceneBuffer", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { shadowPass(passExec); });

    builder->AddGraphicsPass(
        "Geometry Pass",
        [](Pass &pass)
        {
            static VkClearValue colorClearValue{};
            colorClearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};
            colorClearValue.depthStencil = {0.0f};
            pass.AddColorAttachment("position_gbuf", true, &colorClearValue);
            pass.AddColorAttachment("normal_gbuf", true, &colorClearValue);
            pass.AddColorAttachment("albedo_gbuf", true, &colorClearValue);
            pass.AddColorAttachment("metalrough_gbuf", true, &colorClearValue);
            pass.AddDepthStencilAttachment("depth_gbuf", false, &colorClearValue);
            pass.CreatesBuffer("gpuSceneBuffer", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { geometryPass(passExec); });

    builder->AddGraphicsPass(
        "Composite Pass",
        [](Pass &pass)
        {
            static VkClearValue colorClearValue{};
            colorClearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};
            colorClearValue.depthStencil = {0.0f};
            pass.ReadsImage("position_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("normal_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("albedo_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("metalrough_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("shadowMap", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.AddColorAttachment("drawImage", false, &colorClearValue);
            pass.AddDepthStencilAttachment("depth_gbuf", true, nullptr);
            pass.CreatesBuffer("lightBuffer", sizeof(LightData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            pass.CreatesBuffer("gpuSceneBuffer", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { compositePass(passExec); });

    builder->AddGraphicsPass(
        "Transparent Forward Pass",
        [](Pass &pass)
        {
            pass.AddColorAttachment("drawImage", false, nullptr);
            pass.AddDepthStencilAttachment("depth_gbuf", true, nullptr);
            pass.CreatesBuffer("gpuSceneBuffer", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            pass.CreatesBuffer("lightBuffer", sizeof(LightData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { transparentPass(passExec); });
}

void rgraph::DeferredRenderingFeature::geometryPass(rgraph::PassExecution &passExec)
{
    passExec.drawCalls = 0;
    passExec.triangles = 0;

    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(drawContext.OpaqueSurfaces.size());
    for (int i = 0; i < drawContext.OpaqueSurfaces.size(); i++)
    {
        if (is_visible(drawContext.OpaqueSurfaces[i], gpuSceneData.viewproj))
        {
            opaque_draws.push_back(i);
        }
    }

    std::sort(opaque_draws.begin(), opaque_draws.end(),
              [&](const auto &iA, const auto &iB)
              {
                  const RenderObject &A = drawContext.OpaqueSurfaces[iA];
                  const RenderObject &B = drawContext.OpaqueSurfaces[iB];
                  if (A.material == B.material)
                  {
                      return A.indexBuffer < B.indexBuffer;
                  }
                  else
                  {
                      return A.material < B.material;
                  }
              });

    AllocatedBuffer gpuSceneDataBuffer = passExec.allocatedBuffers["gpuSceneBuffer"];
    GPUSceneData *sceneUniformData = (GPUSceneData *)gpuSceneDataBuffer.info.pMappedData;
    *sceneUniformData = gpuSceneData;

    VkDescriptorSet globalDescriptor = passExec.frameDescriptor->allocate(passExec._device, _gpuSceneDataDescriptorLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(passExec._device, globalDescriptor);

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geometryPipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geometryPipeline.layout, 0, 1, &globalDescriptor, 0, nullptr);

    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    MaterialInstance *lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    for (auto &r : opaque_draws)
    {
        const RenderObject &obj = drawContext.OpaqueSurfaces[r];

        if (obj.material != lastMaterial)
        {
            lastMaterial = obj.material;
            vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geometryPipeline.layout, 1, 1, &obj.material->materialSet, 0,
                                    nullptr);
        }

        if (obj.indexBuffer != lastIndexBuffer)
        {
            lastIndexBuffer = obj.indexBuffer;
            vkCmdBindIndexBuffer(passExec.cmd, obj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GPUDrawPushConstants push_constants;
        push_constants.modelMatrix = obj.modelMatrix;
        push_constants.vertexBuffer = obj.vertexBufferAddress;

        vkCmdPushConstants(passExec.cmd, geometryPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(passExec.cmd, obj.indexCount, 1, obj.firstIndex, 0, 0);

        passExec.drawCalls++;
        passExec.triangles += (int)(obj.indexCount / 3);
    }
}

void rgraph::DeferredRenderingFeature::compositePass(rgraph::PassExecution &passExec)
{
    passExec.drawCalls = 0;
    passExec.triangles = 0;

    // Scene data
    AllocatedBuffer gpuSceneDataBuffer = passExec.allocatedBuffers["gpuSceneBuffer"];
    GPUSceneData *sceneUniformData = (GPUSceneData *)gpuSceneDataBuffer.info.pMappedData;
    *sceneUniformData = gpuSceneData;

    VkDescriptorSet sceneDescriptor = passExec.frameDescriptor->allocate(passExec._device, _gpuSceneDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(passExec._device, sceneDescriptor);
    }

    // Light data
    AllocatedBuffer lightAllocBuffer = passExec.allocatedBuffers["lightBuffer"];
    LightData *lightdata = (LightData *)lightAllocBuffer.info.pMappedData;
    lightdata->numLights = drawContext.lights.size();
    for (int i = 0; i < lightdata->numLights; i++)
    {
        PointLight pl = {};
        pl.color = drawContext.lights[i].color;
        pl.transform = drawContext.lights[i].transform;
        pl.intensity = drawContext.lights[i].intensity;
        pl.range = drawContext.lights[i].range;
        lightdata->pointLights[i] = pl;
    }

    VkDescriptorSet lightDescriptor = passExec.frameDescriptor->allocate(passExec._device, lightDescriptorSetLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, lightAllocBuffer.buffer, sizeof(LightData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(passExec._device, lightDescriptor);
    }

    // G-buffer sampler descriptors
    VkDescriptorSet compDescriptor = passExec.frameDescriptor->allocate(passExec._device, compDescriptorSetLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, position_gbuf.imageView, defaultSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, normal_gbuf.imageView, defaultSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(2, albedo_gbuf.imageView, defaultSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(3, metalrough_gbuf.imageView, defaultSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(4, shadowMap.imageView, shadowSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.update_set(passExec._device, compDescriptor);
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline.pipeline);

    VkDescriptorSet sets[] = {compDescriptor, sceneDescriptor, lightDescriptor};
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline.layout, 0, 3, sets, 0, nullptr);

    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    vkCmdDraw(passExec.cmd, 3, 1, 0, 0);

    passExec.drawCalls++;
}

void rgraph::DeferredRenderingFeature::transparentPass(rgraph::PassExecution &passExec)
{
    if (drawContext.TransparentSurfaces.empty())
    {
        return;
    }

    std::vector<uint32_t> transparent_draws;
    transparent_draws.reserve(drawContext.TransparentSurfaces.size());
    for (int i = 0; i < drawContext.TransparentSurfaces.size(); i++)
    {
        if (is_visible(drawContext.TransparentSurfaces[i], gpuSceneData.viewproj))
        {
            transparent_draws.push_back(i);
        }
    }

    // Scene data
    AllocatedBuffer gpuSceneDataBuffer = passExec.allocatedBuffers["gpuSceneBuffer"];
    GPUSceneData *sceneUniformData = (GPUSceneData *)gpuSceneDataBuffer.info.pMappedData;
    *sceneUniformData = gpuSceneData;

    VkDescriptorSet sceneDescriptor = passExec.frameDescriptor->allocate(passExec._device, _gpuSceneDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(passExec._device, sceneDescriptor);
    }

    // Light data
    AllocatedBuffer lightAllocBuffer = passExec.allocatedBuffers["lightBuffer"];
    LightData *lightdata = (LightData *)lightAllocBuffer.info.pMappedData;
    lightdata->numLights = drawContext.lights.size();
    for (int i = 0; i < lightdata->numLights; i++)
    {
        PointLight pl = {};
        pl.color = drawContext.lights[i].color;
        pl.transform = drawContext.lights[i].transform;
        pl.intensity = drawContext.lights[i].intensity;
        pl.range = drawContext.lights[i].range;
        lightdata->pointLights[i] = pl;
    }

    VkDescriptorSet lightDescriptor = passExec.frameDescriptor->allocate(passExec._device, lightDescriptorSetLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, lightAllocBuffer.buffer, sizeof(LightData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(passExec._device, lightDescriptor);
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline.pipeline);

    VkDescriptorSet sets[] = {sceneDescriptor, lightDescriptor};
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline.layout, 0, 2, sets, 0, nullptr);

    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    MaterialInstance *lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    for (auto &idx : transparent_draws)
    {
        auto &r = drawContext.TransparentSurfaces[idx];
        if (r.material != lastMaterial)
        {
            lastMaterial = r.material;
            vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline.layout, 2, 1, &r.material->materialSet, 0,
                                    nullptr);
        }

        if (r.indexBuffer != lastIndexBuffer)
        {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(passExec.cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GPUDrawPushConstants push_constants;
        push_constants.modelMatrix = r.modelMatrix;
        push_constants.vertexBuffer = r.vertexBufferAddress;
        vkCmdPushConstants(passExec.cmd, transparentPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(passExec.cmd, r.indexCount, 1, r.firstIndex, 0, 0);

        passExec.drawCalls++;
        passExec.triangles += (int)(r.indexCount / 3);
    }
}

void rgraph::DeferredRenderingFeature::shadowPass(rgraph::PassExecution &passExec)
{
    passExec.drawCalls = 0;
    passExec.triangles = 0;

    // scene data carries sunViewProj; bind at set 0 (the shadow vertex shader reads it)
    AllocatedBuffer sceneBuf = passExec.allocatedBuffers["shadowSceneBuffer"];
    *(GPUSceneData *)sceneBuf.info.pMappedData = gpuSceneData;

    VkDescriptorSet sceneDescriptor = passExec.frameDescriptor->allocate(passExec._device, _gpuSceneDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, sceneBuf.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(passExec._device, sceneDescriptor);
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.layout, 0, 1, &sceneDescriptor, 0, nullptr);

    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    // draw every opaque surface into the shadow map (no material, no frustum cull)
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
    for (const RenderObject &obj : drawContext.OpaqueSurfaces)
    {
        if (obj.indexBuffer != lastIndexBuffer)
        {
            lastIndexBuffer = obj.indexBuffer;
            vkCmdBindIndexBuffer(passExec.cmd, obj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GPUDrawPushConstants push_constants;
        push_constants.modelMatrix = obj.modelMatrix;
        push_constants.vertexBuffer = obj.vertexBufferAddress;
        vkCmdPushConstants(passExec.cmd, shadowPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(passExec.cmd, obj.indexCount, 1, obj.firstIndex, 0, 0);
        passExec.drawCalls++;
        passExec.triangles += (int)(obj.indexCount / 3);
    }
}

void rgraph::DeferredRenderingFeature::createPipelines(MaterialSystemCreateInfo &materialSystemCreateInfo)
{

    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../shaders/deferred/mrt.frag.spv", materialSystemCreateInfo._device, &meshFragShader))
    {
        fmt::println("Error when building the triangle fragment shader module\n");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../shaders/deferred/mrt.vert.spv", materialSystemCreateInfo._device, &meshVertexShader))
    {
        fmt::println("Error when building the triangle vertex shader module\n");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayout materialLayout = VulkanEngine::Instance().GetMaterialSystem().materialLayout;

    // create pipeline for mrt pass ---------------------------------
    VkDescriptorSetLayout mrtLayouts[] = {materialSystemCreateInfo._gpuSceneDataDescriptorLayout, materialLayout};

    VkPipelineLayoutCreateInfo meshLayoutInfo = vkinit::pipeline_layout_create_info();
    meshLayoutInfo.setLayoutCount = 2;
    meshLayoutInfo.pSetLayouts = mrtLayouts;
    meshLayoutInfo.pPushConstantRanges = &matrixRange;
    meshLayoutInfo.pushConstantRangeCount = 1;

    VkPipelineLayout geomPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(materialSystemCreateInfo._device, &meshLayoutInfo, nullptr, &geomPipelineLayout));

    std::vector<VkFormat> colorAttachmentFormats = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT};

    geometryPipeline.layout = geomPipelineLayout;

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.set_multiple_color_attachments(colorAttachmentFormats);
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    pipelineBuilder.set_depth_format(materialSystemCreateInfo.depthFormat);

    pipelineBuilder._pipelineLayout = geomPipelineLayout;

    geometryPipeline.pipeline = pipelineBuilder.build_pipeline(materialSystemCreateInfo._device);

    vkDestroyShaderModule(materialSystemCreateInfo._device, meshFragShader, nullptr);
    vkDestroyShaderModule(materialSystemCreateInfo._device, meshVertexShader, nullptr);

    // now create pipeline for comp pass -----------------------------

    if (!vkutil::load_shader_module("../shaders/deferred/comp.vert.spv", materialSystemCreateInfo._device, &meshVertexShader))
    {
        fmt::println("Error when building the triangle fragment shader module\n");
    }

    if (!vkutil::load_shader_module("../shaders/deferred/comp.frag.spv", materialSystemCreateInfo._device, &meshFragShader))
    {
        fmt::println("Error when building the triangle fragment shader module\n");
    }

    VkDescriptorSetLayout compLayouts[] = {compDescriptorSetLayout, materialSystemCreateInfo._gpuSceneDataDescriptorLayout, lightDescriptorSetLayout};

    meshLayoutInfo.setLayoutCount = 3;
    meshLayoutInfo.pSetLayouts = compLayouts;
    meshLayoutInfo.pPushConstantRanges = nullptr;
    meshLayoutInfo.pushConstantRangeCount = 0;

    VkPipelineLayout compPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(materialSystemCreateInfo._device, &meshLayoutInfo, nullptr, &compPipelineLayout));

    compositePipeline.layout = compPipelineLayout;

    pipelineBuilder.clear();

    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.set_color_attachment_format(materialSystemCreateInfo.colorFormat);
    pipelineBuilder.disable_blending();
    pipelineBuilder.set_depth_format(materialSystemCreateInfo.depthFormat);
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.disable_depthtest();

    pipelineBuilder._pipelineLayout = compPipelineLayout;

    compositePipeline.pipeline = pipelineBuilder.build_pipeline(materialSystemCreateInfo._device);

    vkDestroyShaderModule(materialSystemCreateInfo._device, meshVertexShader, nullptr);
    vkDestroyShaderModule(materialSystemCreateInfo._device, meshFragShader, nullptr);

    // Transparent forward pipeline -----------------------------

    if (!vkutil::load_shader_module("../shaders/PBR/lights/light_mesh.vert.spv", materialSystemCreateInfo._device, &meshVertexShader))
    {
        fmt::println("Error when building the transparent vertex shader module\n");
    }
    if (!vkutil::load_shader_module("../shaders/PBR/lights/light_mesh.frag.spv", materialSystemCreateInfo._device, &meshFragShader))
    {
        fmt::println("Error when building the transparent fragment shader module\n");
    }

    VkDescriptorSetLayout transparentLayouts[] = {materialSystemCreateInfo._gpuSceneDataDescriptorLayout, lightDescriptorSetLayout, materialLayout};

    meshLayoutInfo.setLayoutCount = 3;
    meshLayoutInfo.pSetLayouts = transparentLayouts;
    meshLayoutInfo.pPushConstantRanges = &matrixRange;
    meshLayoutInfo.pushConstantRangeCount = 1;

    VkPipelineLayout transparentPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(materialSystemCreateInfo._device, &meshLayoutInfo, nullptr, &transparentPipelineLayout));

    transparentPipeline.layout = transparentPipelineLayout;

    pipelineBuilder.clear();
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.set_color_attachment_format(materialSystemCreateInfo.colorFormat);
    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
    pipelineBuilder.set_depth_format(materialSystemCreateInfo.depthFormat);
    pipelineBuilder._pipelineLayout = transparentPipelineLayout;

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(materialSystemCreateInfo._device);

    vkDestroyShaderModule(materialSystemCreateInfo._device, meshVertexShader, nullptr);
    vkDestroyShaderModule(materialSystemCreateInfo._device, meshFragShader, nullptr);

    // Shadow pipeline: depth-only from the sun's view; scene set 0 supplies sunViewProj.
    VkShaderModule shadowVert;
    if (!vkutil::load_shader_module("../shaders/shadow/shadow.vert.spv", materialSystemCreateInfo._device, &shadowVert))
        fmt::println("Error when building the shadow vertex shader\n");
    VkShaderModule shadowFrag;
    if (!vkutil::load_shader_module("../shaders/shadow/shadow.frag.spv", materialSystemCreateInfo._device, &shadowFrag))
        fmt::println("Error when building the shadow fragment shader\n");

    VkDescriptorSetLayout shadowLayouts[] = {materialSystemCreateInfo._gpuSceneDataDescriptorLayout};
    VkPipelineLayoutCreateInfo shadowLayoutInfo = vkinit::pipeline_layout_create_info();
    shadowLayoutInfo.setLayoutCount = 1;
    shadowLayoutInfo.pSetLayouts = shadowLayouts;
    shadowLayoutInfo.pPushConstantRanges = &matrixRange;
    shadowLayoutInfo.pushConstantRangeCount = 1;

    VkPipelineLayout shadowPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(materialSystemCreateInfo._device, &shadowLayoutInfo, nullptr, &shadowPipelineLayout));
    shadowPipeline.layout = shadowPipelineLayout;

    pipelineBuilder.clear();
    pipelineBuilder.set_shaders(shadowVert, shadowFrag);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_LESS_OR_EQUAL); // standard depth
    pipelineBuilder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    pipelineBuilder._pipelineLayout = shadowPipelineLayout;
    shadowPipeline.pipeline = pipelineBuilder.build_pipeline(materialSystemCreateInfo._device);

    vkDestroyShaderModule(materialSystemCreateInfo._device, shadowVert, nullptr);
    vkDestroyShaderModule(materialSystemCreateInfo._device, shadowFrag, nullptr);
}

void rgraph::DeferredRenderingFeature::createImages(DeletionQueue &delQueue)
{
    auto windowExtent = VulkanEngine::Instance().GetWindowExtent();
    auto device = VulkanEngine::Instance().GetVkDevice();
    VkExtent3D imageExtent = {windowExtent.width, windowExtent.height, 1};

    std::vector<AllocatedImage *> gbufImages = {&position_gbuf, &normal_gbuf, &albedo_gbuf, &metalrough_gbuf};
    std::vector<std::string> gbufNames = {"position_gbuf", "normal_gbuf", "albedo_gbuf", "metalrough_gbuf"};

    Rendergraph &rgraphInstance = Rendergraph::Instance();

    // create the color attachments
    for (int i = 0; i < gbufImages.size(); i++)
    {
        auto &gbufImage = gbufImages[i];
        gbufImage->imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        gbufImage->imageExtent = imageExtent;

        VkImageUsageFlags colorImageUses{};
        colorImageUses |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        colorImageUses |= VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImageCreateInfo rimg_info = vkinit::image_create_info(gbufImage->imageFormat, colorImageUses, imageExtent);

        VmaAllocationCreateInfo rimg_allocinfo = {};
        rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        GPUResourceAllocator &_gpuResourceAllocator = GPUResourceAllocator::Instance();
        _gpuResourceAllocator.create_image(&rimg_info, &rimg_allocinfo, &gbufImage->image, &gbufImage->allocation, nullptr);

        VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(gbufImage->imageFormat, gbufImage->image, VK_IMAGE_ASPECT_COLOR_BIT);

        VK_CHECK(vkCreateImageView(device, &rview_info, nullptr, &gbufImage->imageView));

        rgraphInstance.AddTrackedImage(gbufNames[i], VK_IMAGE_LAYOUT_UNDEFINED, *gbufImage);

        delQueue.push_function(
            [=, this]
            {
                auto &_gpuResourceAllocator = GPUResourceAllocator::Instance();
                vkDestroyImageView(device, gbufImage->imageView, nullptr);
                _gpuResourceAllocator.destroy_image(gbufImage->image, gbufImage->allocation);
            });
    }

    // create the depth-stencil attachment

    depth_gbuf.imageFormat = VK_FORMAT_D32_SFLOAT;
    depth_gbuf.imageExtent = imageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(depth_gbuf.imageFormat, depthImageUsages, imageExtent);

    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    GPUResourceAllocator::Instance().create_image(&dimg_info, &rimg_allocinfo, &depth_gbuf.image, &depth_gbuf.allocation, nullptr);

    // build a image-view for the depth image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(depth_gbuf.imageFormat, depth_gbuf.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(device, &dview_info, nullptr, &depth_gbuf.imageView));

    rgraphInstance.AddTrackedImage("depth_gbuf", VK_IMAGE_LAYOUT_UNDEFINED, depth_gbuf);

    delQueue.push_function(
        [=, this]
        {
            auto &_gpuResourceAllocator = GPUResourceAllocator::Instance();
            vkDestroyImageView(device, depth_gbuf.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(depth_gbuf.image, depth_gbuf.allocation);
        });

    // shadow map: like depth_gbuf but also SAMPLED so the composite can read it.
    shadowMap.imageFormat = VK_FORMAT_D32_SFLOAT;
    shadowMap.imageExtent = imageExtent;
    VkImageUsageFlags shadowUsages = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImageCreateInfo simg_info = vkinit::image_create_info(shadowMap.imageFormat, shadowUsages, imageExtent);

    GPUResourceAllocator::Instance().create_image(&simg_info, &rimg_allocinfo, &shadowMap.image, &shadowMap.allocation, nullptr);

    VkImageViewCreateInfo sview_info = vkinit::imageview_create_info(shadowMap.imageFormat, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(device, &sview_info, nullptr, &shadowMap.imageView));

    rgraphInstance.AddTrackedImage("shadowMap", VK_IMAGE_LAYOUT_UNDEFINED, shadowMap);

    delQueue.push_function(
        [=, this]
        {
            auto &_gpuResourceAllocator = GPUResourceAllocator::Instance();
            vkDestroyImageView(device, shadowMap.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(shadowMap.image, shadowMap.allocation);
        });
}