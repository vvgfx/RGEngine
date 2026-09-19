#include "SSRFeature.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <cmath>

rgraph::SSRFeature::SSRFeature(VkDevice device, DeletionQueue &delQueue, GPUSceneData &sceneData) : sceneData(sceneData)
{
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // lit scene
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // output
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // position
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // normal
        builder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // metal/rough
        builder.add_binding(5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // scene
        descriptorLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // The march samples between texels when refining a hit, so this filters linearly.
    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_LINEAR,
                                    .minFilter = VK_FILTER_LINEAR,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));

    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .setLayoutCount = 1,
                                          .pSetLayouts = &descriptorLayout,
                                          .pushConstantRangeCount = 1,
                                          .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule module;
    if (!vkutil::load_shader_module("../shaders/ssr/ssr.comp.spv", device, &module))
    {
        fmt::println("SSR: failed to load ssr.comp.spv");
    }
    else
    {
        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"};
        VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = pipelineLayout};
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        vkDestroyShaderModule(device, module, nullptr);
    }

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroySampler(device, sampler, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        });
}

void rgraph::SSRFeature::Register(rgraph::Rendergraph *builder)
{
    if (pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddComputePass(
        "ssr",
        [](Pass &pass)
        {
            pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("position_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("normal_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("metalrough_gbuf", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.WritesImage("sceneImage");
            pass.CreatesBuffer("ssrScene", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        },
        [&](PassExecution &passExec) { run(passExec); });
}

void rgraph::SSRFeature::run(PassExecution &passExec)
{
    AllocatedBuffer sceneBuffer = passExec.allocatedBuffers["ssrScene"];
    *(GPUSceneData *)sceneBuffer.info.pMappedData = sceneData;

    // PassExecution carries every tracked image, so the G-buffers need no constructor plumbing.
    auto image = [&](const char *name) { return passExec.allocatedImages[name]; };

    VkDescriptorSet set = passExec.frameDescriptor->allocate(passExec._device, descriptorLayout);
    DescriptorWriter writer;
    writer.write_image(0, image("drawImage").imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, image("sceneImage").imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.write_image(2, image("position_gbuf").imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, image("normal_gbuf").imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(4, image("metalrough_gbuf").imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_buffer(5, sceneBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(passExec._device, set);

    PushConstants push{glm::vec4(settings.enabled ? 1.0f : 0.0f, settings.maxRoughness, float(settings.steps), settings.thickness),
                       glm::vec4(settings.maxDistance, settings.intensity, settings.showReflectionOnly ? 1.0f : 0.0f, 0.0f)};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    const uint32_t groupsX = uint32_t(std::ceil(passExec._drawExtent.width / 16.0));
    const uint32_t groupsY = uint32_t(std::ceil(passExec._drawExtent.height / 16.0));
    vkCmdDispatch(passExec.cmd, groupsX, groupsY, 1);

    passExec.dispatchCalls = float(groupsX * groupsY);
}
