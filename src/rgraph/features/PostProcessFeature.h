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
        PostProcessFeature(VkDevice device, DeletionQueue &delQueue, AllocatedImage drawImage, AllocatedImage postImage);

        void Register(Rendergraph *builder) override;

        PostSettings settings;

      private:
        void run(PassExecution &passExec);

        struct PushConstants
        {
            glm::vec4 params;
        };

        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        DescriptorAllocatorGrowable descriptorAllocator;
    };
} // namespace rgraph
