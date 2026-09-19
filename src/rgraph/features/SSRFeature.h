#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    struct SSRSettings
    {
        bool enabled = true;
        // Bistro's spec-gloss materials convert to roughness ~0.19 for things like awning fabric,
        // which is far too rough for a single sharp mirror ray. Only near-glass surfaces qualify.
        float maxRoughness = 0.25f;
        int steps = 24;
        float thickness = 0.25f;  // world units; how deep a surface is assumed to be
        float maxDistance = 30.f; // world units a ray may travel
        float intensity = 1.0f;
        bool showReflectionOnly = false; // debug: output the reflection term alone
    };

    /**
     * @brief Screen-space reflections.
     *
     * Writes `sceneImage = lit + reflection` rather than compositing back into drawImage: the march
     * reads drawImage at arbitrary texels (wherever the ray lands), so writing the same image in the
     * same dispatch would be a race. Downstream passes read sceneImage instead, which also means
     * reflections bloom and tonemap like everything else.
     *
     * The pass always runs, copying the lit colour through when disabled, so the post chain never
     * has to care whether reflections happened.
     */
    class SSRFeature : public IFeature
    {
      public:
        SSRFeature(VkDevice device, DeletionQueue &delQueue, GPUSceneData &sceneData);

        void Register(Rendergraph *builder) override;

        SSRSettings settings;

      private:
        void run(PassExecution &passExec);

        struct PushConstants
        {
            glm::vec4 params;
            glm::vec4 params2;
        };

        GPUSceneData &sceneData;

        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
    };
} // namespace rgraph
