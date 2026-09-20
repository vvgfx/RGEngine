#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <string>

namespace rgraph
{
    struct PostSettings
    {
        bool fxaa = true;
        float exposureEV = 0.0f; // stops

        float bloomIntensity = 0.5f;
        float bloomThreshold = 1.5f;

        /// Ceiling on the bright pass. The blur is narrow, so an extreme value becomes a tight
        /// blazing fringe rather than a glow -- an HDRI sun reaches 75,000. Standard firefly clamp.
        float bloomClamp = 8.0f; // HDR luminance above which light starts to bleed

        // set while a debug view is active: skip tonemap, gamma and bloom
        bool passthrough = false;
    };

    /**
     * @brief Resolves the linear HDR draw image to a tonemapped, anti-aliased image.
     *
     * Tonemapping and FXAA are separate passes on purpose: folding them together meant FXAA ran the
     * ACES curve and a pow for every one of its ~13 taps.
     */
    class PostProcessFeature : public IFeature
    {
      public:
        /// hdrName must be the rendergraph name of hdrImage: the pass declarations drive layout
        /// transitions, so naming the wrong image would transition something the shaders never read.
        PostProcessFeature(VkDevice device, DeletionQueue &delQueue, std::string hdrName, AllocatedImage hdrImage, AllocatedImage postImage,
                           AllocatedImage ldrImage, AllocatedImage bloomA, AllocatedImage bloomB);

        void Register(Rendergraph *builder) override;

        PostSettings settings;

      private:
        struct PushConstants
        {
            glm::vec4 params;
        };

        void dispatch(PassExecution &passExec, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set, glm::vec4 params,
                      VkExtent3D extent);

        void runBloom(PassExecution &passExec, VkPipeline target, VkDescriptorSet set, glm::vec4 params);
        void runTonemap(PassExecution &passExec);
        void runFxaa(PassExecution &passExec);

        // extract, blur and FXAA share one shape: sampled source -> storage destination
        VkDescriptorSetLayout blitLayout = VK_NULL_HANDLE;
        VkPipelineLayout blitPipelineLayout = VK_NULL_HANDLE;
        VkPipeline extractPipeline = VK_NULL_HANDLE;
        VkPipeline blurPipeline = VK_NULL_HANDLE;
        VkPipeline fxaaPipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout tonemapLayout = VK_NULL_HANDLE;
        VkPipelineLayout tonemapPipelineLayout = VK_NULL_HANDLE;
        VkPipeline tonemapPipeline = VK_NULL_HANDLE;

        VkDescriptorSet setExtract = VK_NULL_HANDLE; // drawImage -> A
        VkDescriptorSet setAB = VK_NULL_HANDLE;      // A -> B
        VkDescriptorSet setBA = VK_NULL_HANDLE;      // B -> A
        VkDescriptorSet setTonemap = VK_NULL_HANDLE;
        VkDescriptorSet setFxaa = VK_NULL_HANDLE;

        VkSampler sampler = VK_NULL_HANDLE;
        std::string hdrName;
        VkExtent3D fullExtent{};
        VkExtent3D bloomExtent{};

        DescriptorAllocatorGrowable descriptorAllocator;
    };
} // namespace rgraph
