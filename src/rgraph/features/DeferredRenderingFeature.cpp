#include "rgraph/features/DeferredRenderingFeature.h"
#include "DeferredRenderingFeature.h"
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
        compDescriptorSetLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Declare the G-buffer images as transient resources owned by the rendergraph.
    // The rendergraph's Build() will allocate them with memory aliasing on the first frame.
    auto windowExtent = VulkanEngine::Instance().GetWindowExtent();
    VkExtent3D imageExtent = {windowExtent.width, windowExtent.height, 1};

    VkImageUsageFlags colorUse = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImageUsageFlags depthUse = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    Rendergraph &rg = Rendergraph::Instance();
    rg.DeclareTransientImage("position_gbuf", imageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, colorUse);
    rg.DeclareTransientImage("normal_gbuf", imageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, colorUse);
    rg.DeclareTransientImage("albedo_gbuf", imageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, colorUse);
    rg.DeclareTransientImage("metalrough_gbuf", imageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, colorUse);
    rg.DeclareTransientImage("depth_gbuf", imageExtent, VK_FORMAT_D32_SFLOAT, depthUse);

    createPipelines(materialSystemCreateInfo);

    delQueue.push_function(
        [_device, this]()
        {
            vkDestroySampler(_device, defaultSampler, nullptr);
            vkDestroyDescriptorSetLayout(_device, lightDescriptorSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, compDescriptorSetLayout, nullptr);
            vkDestroyPipelineLayout(_device, geometryPipeline.layout, nullptr);
            vkDestroyPipelineLayout(_device, compositePipeline.layout, nullptr);
            vkDestroyPipelineLayout(_device, transparentPipeline.layout, nullptr);
            vkDestroyPipeline(_device, geometryPipeline.pipeline, nullptr);
            vkDestroyPipeline(_device, compositePipeline.pipeline, nullptr);
            vkDestroyPipeline(_device, transparentPipeline.pipeline, nullptr);
        });
}

void rgraph::DeferredRenderingFeature::Register(rgraph::Rendergraph *builder)
{

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

    // G-buffer sampler descriptors — read image views from the rendergraph's image table.
    VkDescriptorSet compDescriptor = passExec.frameDescriptor->allocate(passExec._device, compDescriptorSetLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, passExec.allocatedImages.at("position_gbuf").imageView, defaultSampler,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, passExec.allocatedImages.at("normal_gbuf").imageView, defaultSampler,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(2, passExec.allocatedImages.at("albedo_gbuf").imageView, defaultSampler,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(3, passExec.allocatedImages.at("metalrough_gbuf").imageView, defaultSampler,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
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
}

