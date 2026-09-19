#include "PostProcessFeature.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <cmath>
#include <string_view>

namespace
{
    VkPipeline createComputePipeline(VkDevice device, const char *path, VkPipelineLayout layout)
    {
        VkShaderModule module;
        if (!vkutil::load_shader_module(path, device, &module))
        {
            fmt::println("PostProcess: failed to load {}", path);
            return VK_NULL_HANDLE;
        }

        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"};
        VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = layout};

        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        vkDestroyShaderModule(device, module, nullptr);
        return pipeline;
    }
} // namespace

rgraph::PostProcessFeature::PostProcessFeature(VkDevice device, DeletionQueue &delQueue, AllocatedImage drawImage, AllocatedImage postImage,
                                               AllocatedImage bloomA, AllocatedImage bloomB)
{
    bloomExtent = bloomA.imageExtent;

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    descriptorAllocator.init(device, 10, sizes);

    // Extract and both blur directions share one shape, so they share one layout and one pipeline
    // layout; only the bound set and the push constants differ.
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        blitLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // linear HDR source
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // resolved output
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // bloom
        postLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // FXAA and the blur both read between texels, so this must filter linearly
    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                    .magFilter = VK_FILTER_LINEAR,
                                    .minFilter = VK_FILTER_LINEAR,
                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));

    auto writeBlit = [&](const AllocatedImage &src, const AllocatedImage &dst)
    {
        VkDescriptorSet set = descriptorAllocator.allocate(device, blitLayout);
        DescriptorWriter writer;
        writer.write_image(0, src.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, dst.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(device, set);
        return set;
    };

    // All images are persistent, so these sets are written once.
    setExtract = writeBlit(drawImage, bloomA);
    setAB = writeBlit(bloomA, bloomB);
    setBA = writeBlit(bloomB, bloomA);

    descriptorSet = descriptorAllocator.allocate(device, postLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, drawImage.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, postImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.write_image(2, bloomA.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.update_set(device, descriptorSet);
    }

    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};

    VkPipelineLayoutCreateInfo postInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1,
                                        .pSetLayouts = &postLayout,
                                        .pushConstantRangeCount = 1,
                                        .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &postInfo, nullptr, &pipelineLayout));

    VkPipelineLayoutCreateInfo blitInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1,
                                        .pSetLayouts = &blitLayout,
                                        .pushConstantRangeCount = 1,
                                        .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &blitInfo, nullptr, &blitPipelineLayout));

    pipeline = createComputePipeline(device, "../shaders/post/post.comp.spv", pipelineLayout);
    extractPipeline = createComputePipeline(device, "../shaders/post/bloom_extract.comp.spv", blitPipelineLayout);
    blurPipeline = createComputePipeline(device, "../shaders/post/bloom_blur.comp.spv", blitPipelineLayout);

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipeline(device, extractPipeline, nullptr);
            vkDestroyPipeline(device, blurPipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyPipelineLayout(device, blitPipelineLayout, nullptr);
            vkDestroySampler(device, sampler, nullptr);
            descriptorAllocator.destroy_pools(device);
            vkDestroyDescriptorSetLayout(device, postLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, blitLayout, nullptr);
        });
}

void rgraph::PostProcessFeature::Register(rgraph::Rendergraph *builder)
{
    if (pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    const bool bloomOn = extractPipeline != VK_NULL_HANDLE && blurPipeline != VK_NULL_HANDLE;

    if (bloomOn)
    {
        builder->AddComputePass(
            "bloom-extract",
            [](Pass &pass)
            {
                pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pass.WritesImage("bloomA");
            },
            [&](PassExecution &passExec)
            { runBloom(passExec, extractPipeline, setExtract, glm::vec4(settings.bloomThreshold, settings.bloomThreshold * 0.5f, 0.0f, 0.0f)); });

        // Two horizontal/vertical pairs with increasing radius: a much smoother falloff than one
        // pass, for four extra half-res dispatches.
        struct BlurStep
        {
            const char *name;
            const char *src;
            const char *dst;
            glm::vec2 dir;
            float scale;
        };
        static const BlurStep steps[] = {
            {"bloom-blur-h", "bloomA", "bloomB", {1, 0}, 1.0f},
            {"bloom-blur-v", "bloomB", "bloomA", {0, 1}, 1.0f},
            {"bloom-blur-h2", "bloomA", "bloomB", {1, 0}, 2.0f},
            {"bloom-blur-v2", "bloomB", "bloomA", {0, 1}, 2.0f},
        };

        for (const BlurStep &step : steps)
        {
            VkDescriptorSet set = step.dst == std::string_view("bloomB") ? setAB : setBA;
            glm::vec4 params(step.dir.x, step.dir.y, settings.bloomRadius * step.scale, 0.0f);

            builder->AddComputePass(
                step.name,
                [&step](Pass &pass)
                {
                    pass.ReadsImage(step.src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pass.WritesImage(step.dst);
                },
                [this, set, params](PassExecution &passExec) { runBloom(passExec, blurPipeline, set, params); });
        }
    }

    builder->AddComputePass(
        "post-process",
        [bloomOn](Pass &pass)
        {
            pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (bloomOn)
            {
                pass.ReadsImage("bloomA", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            pass.WritesImage("postImage");
        },
        [&](PassExecution &passExec) { run(passExec); });
}

void rgraph::PostProcessFeature::runBloom(PassExecution &passExec, VkPipeline target, VkDescriptorSet set, glm::vec4 params)
{
    // The pass is still declared so its layout transitions happen; only the work is skipped.
    if (settings.passthrough || settings.bloomIntensity <= 0.0f)
    {
        return;
    }

    PushConstants push{params};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blitPipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, blitPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    const uint32_t groupsX = uint32_t(std::ceil(bloomExtent.width / 16.0));
    const uint32_t groupsY = uint32_t(std::ceil(bloomExtent.height / 16.0));
    vkCmdDispatch(passExec.cmd, groupsX, groupsY, 1);

    passExec.dispatchCalls = float(groupsX * groupsY);
}

void rgraph::PostProcessFeature::run(PassExecution &passExec)
{
    const float bloom = settings.passthrough ? 0.0f : settings.bloomIntensity;

    PushConstants push{glm::vec4(settings.exposure, settings.fxaa && !settings.passthrough ? 1.0f : 0.0f,
                                 settings.passthrough ? 1.0f : 0.0f, bloom)};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    const uint32_t groupsX = uint32_t(std::ceil(passExec._drawExtent.width / 16.0));
    const uint32_t groupsY = uint32_t(std::ceil(passExec._drawExtent.height / 16.0));
    vkCmdDispatch(passExec.cmd, groupsX, groupsY, 1);

    passExec.dispatchCalls = float(groupsX * groupsY);
}
