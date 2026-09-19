#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    struct PostSettings
    {
        bool fxaa = true;
        float exposure = 1.0f;

        float bloomIntensity = 0.6f;
        float bloomThreshold = 2.0f; // HDR luminance; tuned against emissive vs sunlit surfaces
        float bloomRadius = 1.0f;

        // set while a debug view is active: skip exposure, tonemap, gamma and bloom
        bool passthrough = false;
    };

    /**
     * @brief Resolves the linear HDR draw image to a tonemapped, gamma-encoded, anti-aliased image.
     *
     * Shading passes now emit linear HDR so additive transparency blends correctly; tonemapping and
     * gamma live here, at the end, applied exactly once.
     */
    class PostProcessFeature : public IFeature
    {
      public:
        PostProcessFeature(VkDevice device, DeletionQueue &delQueue, AllocatedImage drawImage, AllocatedImage postImage, AllocatedImage bloomA,
                           AllocatedImage bloomB);

        void Register(Rendergraph *builder) override;

        PostSettings settings;

      private:
        void run(PassExecution &passExec);
        void runBloom(PassExecution &passExec, VkPipeline target, VkDescriptorSet set, glm::vec4 params);

        struct PushConstants
        {
            glm::vec4 params;
        };

        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout postLayout = VK_NULL_HANDLE; // hdr + out + bloom
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

        // extract and blur share one shape: sampled source -> storage destination
        VkDescriptorSetLayout blitLayout = VK_NULL_HANDLE;
        VkPipeline extractPipeline = VK_NULL_HANDLE;
        VkPipeline blurPipeline = VK_NULL_HANDLE;
        VkPipelineLayout blitPipelineLayout = VK_NULL_HANDLE;

        VkDescriptorSet setExtract = VK_NULL_HANDLE; // drawImage -> A
        VkDescriptorSet setAB = VK_NULL_HANDLE;      // A -> B
        VkDescriptorSet setBA = VK_NULL_HANDLE;      // B -> A

        VkSampler sampler = VK_NULL_HANDLE;
        VkExtent3D bloomExtent{};
        DescriptorAllocatorGrowable descriptorAllocator;
    };
} // namespace rgraph
