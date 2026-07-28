#include "SkyboxFeature.h"
#include "rgraph/Rendergraph.h"
#include "vk_descriptors.h"
#include "vk_pipelines.h"
#include <cstring>
#include <vulkan/vulkan_core.h>

using namespace rgraph;

SkyboxFeature::SkyboxFeature(VkDevice device, VkFormat colorFormat, VkFormat depthFormat, DeletionQueue &delQueue)
{
    {
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        skyUboLayout = b.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    {
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        b.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        inputLayout = b.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        VkSamplerCreateInfo si{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &si, nullptr, &sampler);
    }

    VkShaderModule vert{}, frag{};
    if (!vkutil::load_shader_module("../shaders/sky/skybox.vert.spv", device, &vert))
        fmt::println("SkyboxFeature: failed to load skybox.vert.spv");
    if (!vkutil::load_shader_module("../shaders/sky/skybox.frag.spv", device, &frag))
        fmt::println("SkyboxFeature: failed to load skybox.frag.spv");

    VkDescriptorSetLayout layouts[] = {skyUboLayout, inputLayout};
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = layouts;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipeline.layout));

    PipelineBuilder builder;
    builder.set_shaders(vert, frag);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    builder.set_multisampling_none();
    builder.set_color_attachment_format(colorFormat);
    builder.disable_blending();
    builder.set_depth_format(depthFormat);
    builder.disable_depthtest(); // background masking is done in the shader via the normal G-buffer
    builder._pipelineLayout = pipeline.layout;
    pipeline.pipeline = builder.build_pipeline(device);

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);

    delQueue.push_function(
        [device, this]()
        {
            vkDestroySampler(device, sampler, nullptr);
            vkDestroyDescriptorSetLayout(device, skyUboLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, inputLayout, nullptr);
            vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        });
}

void SkyboxFeature::Register(Rendergraph *builder)
{
    if (!enabled)
        return;

    builder->AddGraphicsPass(
        "Skybox",
        [this](Pass &pass)
        {
            pass.AddColorAttachment("drawImage", false, nullptr); // LOAD: preserve the lit scene
            pass.AddDepthStencilAttachment("depth_gbuf", false, nullptr);
            pass.ReadsImage("normal_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.CreatesBuffer("skyParamsBuffer", sizeof(Params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [this](PassExecution &passExec) { draw(passExec); });
}

void SkyboxFeature::draw(PassExecution &passExec)
{
    AllocatedBuffer paramsBuf = passExec.allocatedBuffers["skyParamsBuffer"];
    std::memcpy(paramsBuf.info.pMappedData, &params, sizeof(Params));
    VkDescriptorSet set0 = passExec.frameDescriptor->allocate(passExec._device, skyUboLayout);
    {
        DescriptorWriter w;
        w.write_buffer(0, paramsBuf.buffer, sizeof(Params), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        w.update_set(passExec._device, set0);
    }

    AllocatedImage normalImg = passExec.allocatedImages["normal_gbuf"];
    VkDescriptorSet set1 = passExec.frameDescriptor->allocate(passExec._device, inputLayout);
    {
        DescriptorWriter w;
        w.write_image(0, normalImg.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        w.write_image(1, skyTexture.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        w.update_set(passExec._device, set1);
    }

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    VkDescriptorSet sets[] = {set0, set1};
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 2, sets, 0, nullptr);

    VkViewport viewport{};
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    vkCmdDraw(passExec.cmd, 3, 1, 0, 0); // fullscreen triangle

    passExec.drawCalls = 1;
    passExec.triangles = 1;
}
