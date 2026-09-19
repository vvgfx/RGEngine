#include "PostProcessFeature.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <cmath>

rgraph::PostProcessFeature::PostProcessFeature(VkDevice device, DeletionQueue &delQueue, AllocatedImage drawImage, AllocatedImage postImage)
{
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    descriptorAllocator.init(device, 10, sizes);

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // linear HDR source
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // resolved output
        descriptorLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // FXAA reads between texels, so this must filter linearly
    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_LINEAR,
                                    .minFilter = VK_FILTER_LINEAR,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));

    descriptorSet = descriptorAllocator.allocate(device, descriptorLayout);

    // Both images are persistent and the graph keeps drawImage readable, so this set is written once.
    DescriptorWriter writer;
    writer.write_image(0, drawImage.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, postImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(device, descriptorSet);

    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .setLayoutCount = 1,
                                          .pSetLayouts = &descriptorLayout,
                                          .pushConstantRangeCount = 1,
                                          .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule module;
    if (!vkutil::load_shader_module("../shaders/post/post.comp.spv", device, &module))
    {
        fmt::println("PostProcess: failed to load post.comp.spv");
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
            descriptorAllocator.destroy_pools(device);
            vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        });
}

void rgraph::PostProcessFeature::Register(rgraph::Rendergraph *builder)
{
    if (pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddComputePass(
        "post-process",
        [](Pass &pass)
        {
            pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.WritesImage("postImage");
        },
        [&](PassExecution &passExec) { run(passExec); });
}

void rgraph::PostProcessFeature::run(PassExecution &passExec)
{
    PushConstants push{glm::vec4(settings.exposure, settings.fxaa && !settings.passthrough ? 1.0f : 0.0f,
                                 settings.passthrough ? 1.0f : 0.0f, 0.0f)};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    const uint32_t groupsX = uint32_t(std::ceil(passExec._drawExtent.width / 16.0));
    const uint32_t groupsY = uint32_t(std::ceil(passExec._drawExtent.height / 16.0));
    vkCmdDispatch(passExec.cmd, groupsX, groupsY, 1);

    passExec.dispatchCalls = float(groupsX * groupsY);
}
