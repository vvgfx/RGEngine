#include "PostProcessFeature.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <cmath>
#include <utility>

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

    VkPipelineLayout createLayout(VkDevice device, VkDescriptorSetLayout setLayout, uint32_t pushSize)
    {
        VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pushSize};
        VkPipelineLayoutCreateInfo info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1,
                                        .pSetLayouts = &setLayout,
                                        .pushConstantRangeCount = 1,
                                        .pPushConstantRanges = &range};
        VkPipelineLayout layout;
        VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
        return layout;
    }
} // namespace

rgraph::PostProcessFeature::PostProcessFeature(VkDevice device, DeletionQueue &delQueue, std::string hdrName, AllocatedImage hdrImage,
                                               AllocatedImage postImage, AllocatedImage ldrImage, AllocatedImage bloomA, AllocatedImage bloomB)
    : hdrName(std::move(hdrName))
{
    fullExtent = hdrImage.imageExtent;
    bloomExtent = bloomA.imageExtent;

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                                                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    descriptorAllocator.init(device, 10, sizes);

    // Extract, blur and FXAA all share one shape, so they share a layout and a pipeline layout.
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        blitLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // linear HDR
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // LDR output
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // bloom
        tonemapLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // FXAA and the bloom blur both read between texels, so this must filter linearly
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

    // Every image here is persistent, so these sets are written once.
    setExtract = writeBlit(hdrImage, bloomA);
    setAB = writeBlit(bloomA, bloomB);
    setBA = writeBlit(bloomB, bloomA);
    setFxaa = writeBlit(ldrImage, postImage);

    setTonemap = descriptorAllocator.allocate(device, tonemapLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, hdrImage.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, ldrImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.write_image(2, bloomA.imageView, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.update_set(device, setTonemap);
    }

    blitPipelineLayout = createLayout(device, blitLayout, sizeof(PushConstants));
    tonemapPipelineLayout = createLayout(device, tonemapLayout, sizeof(PushConstants));

    extractPipeline = createComputePipeline(device, "../shaders/post/bloom_extract.comp.spv", blitPipelineLayout);
    blurPipeline = createComputePipeline(device, "../shaders/post/bloom_blur.comp.spv", blitPipelineLayout);
    fxaaPipeline = createComputePipeline(device, "../shaders/post/fxaa.comp.spv", blitPipelineLayout);
    tonemapPipeline = createComputePipeline(device, "../shaders/post/tonemap.comp.spv", tonemapPipelineLayout);

    delQueue.push_function(
        [device, this]()
        {
            for (VkPipeline p : {extractPipeline, blurPipeline, fxaaPipeline, tonemapPipeline})
            {
                vkDestroyPipeline(device, p, nullptr);
            }
            vkDestroyPipelineLayout(device, blitPipelineLayout, nullptr);
            vkDestroyPipelineLayout(device, tonemapPipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, blitLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, tonemapLayout, nullptr);
            vkDestroySampler(device, sampler, nullptr);
            descriptorAllocator.destroy_pools(device);
        });
}

void rgraph::PostProcessFeature::Register(rgraph::Rendergraph *builder)
{
    if (tonemapPipeline == VK_NULL_HANDLE || fxaaPipeline == VK_NULL_HANDLE)
    {
        return;
    }

    const bool bloomOn = extractPipeline != VK_NULL_HANDLE && blurPipeline != VK_NULL_HANDLE;

    if (bloomOn)
    {
        builder->AddComputePass(
            "bloom-extract",
            [this](Pass &pass)
            {
                pass.ReadsImage(hdrName, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pass.WritesImage("bloomA");
            },
            [&](PassExecution &passExec) { runBloom(passExec, extractPipeline, setExtract, glm::vec4(settings.bloomThreshold, settings.bloomClamp, 0, 0)); });

        builder->AddComputePass(
            "bloom-blur-h",
            [](Pass &pass)
            {
                pass.ReadsImage("bloomA", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pass.WritesImage("bloomB");
            },
            [&](PassExecution &passExec) { runBloom(passExec, blurPipeline, setAB, glm::vec4(1, 0, 1, 0)); });

        builder->AddComputePass(
            "bloom-blur-v",
            [](Pass &pass)
            {
                pass.ReadsImage("bloomB", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pass.WritesImage("bloomA");
            },
            [&](PassExecution &passExec) { runBloom(passExec, blurPipeline, setBA, glm::vec4(0, 1, 1, 0)); });
    }

    builder->AddComputePass(
        "tonemap",
        [this, bloomOn](Pass &pass)
        {
            pass.ReadsImage(hdrName, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (bloomOn)
            {
                pass.ReadsImage("bloomA", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            pass.WritesImage("ldrImage");
        },
        [&](PassExecution &passExec) { runTonemap(passExec); });

    builder->AddComputePass(
        "fxaa",
        [](Pass &pass)
        {
            pass.ReadsImage("ldrImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.WritesImage("postImage");
        },
        [&](PassExecution &passExec) { runFxaa(passExec); });
}

void rgraph::PostProcessFeature::dispatch(PassExecution &passExec, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set,
                                          glm::vec4 params, VkExtent3D extent)
{
    PushConstants push{params};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(passExec.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    const uint32_t groupsX = uint32_t(std::ceil(extent.width / 16.0));
    const uint32_t groupsY = uint32_t(std::ceil(extent.height / 16.0));
    vkCmdDispatch(passExec.cmd, groupsX, groupsY, 1);

    passExec.dispatchCalls = float(groupsX * groupsY);
}

void rgraph::PostProcessFeature::runBloom(PassExecution &passExec, VkPipeline target, VkDescriptorSet set, glm::vec4 params)
{
    // The pass is still declared so its layout transitions happen; only the work is skipped.
    if (settings.passthrough || settings.bloomIntensity <= 0.0f)
    {
        return;
    }

    dispatch(passExec, target, blitPipelineLayout, set, params, bloomExtent);
}

void rgraph::PostProcessFeature::runTonemap(PassExecution &passExec)
{
    const float bloom = settings.passthrough ? 0.0f : settings.bloomIntensity;

    dispatch(passExec, tonemapPipeline, tonemapPipelineLayout, setTonemap,
             glm::vec4(settings.exposureEV, 0.0f, settings.passthrough ? 1.0f : 0.0f, bloom), fullExtent);
}

void rgraph::PostProcessFeature::runFxaa(PassExecution &passExec)
{
    dispatch(passExec, fxaaPipeline, blitPipelineLayout, setFxaa,
             glm::vec4(0.0f, settings.fxaa && !settings.passthrough ? 1.0f : 0.0f, 0.0f, 0.0f), fullExtent);
}
