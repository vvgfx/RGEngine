#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    struct ShadowSettings
    {
        bool enabled = true;
        float pcfRadius = 1.0f;
        float cascadeSplitLambda = 0.85f; // 0 = uniform splits, 1 = fully logarithmic
        float maxDistance = 120.0f;       // world units the cascades need to cover
    };

    /// Matches ShadowData in shadow_input.glsl.
    struct ShadowDataGPU
    {
        glm::mat4 cascadeViewProj[4];
        glm::vec4 cascadeSplits;  // view-space far distance of each cascade
        glm::vec4 params;         // x = texel world size, y = pcf radius, z = atlas dim, w = enabled
    };

    /**
     * @brief Cascaded shadow maps for the single directional light.
     *
     * Four cascades packed into one depth atlas, so there is a single image, a single sampler and
     * one descriptor for the composite pass to bind. Cascades are fitted to slices of the camera
     * frustum and snapped to texel boundaries so they do not shimmer as the camera moves.
     */
    class ShadowFeature : public IFeature
    {
      public:
        static constexpr uint32_t CASCADE_COUNT = 4;
        static constexpr uint32_t CASCADE_RES = 2048;
        static constexpr uint32_t ATLAS_DIM = CASCADE_RES * 2; // 2x2 layout

        ShadowFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData);

        void Register(Rendergraph *builder) override;

        VkDescriptorSetLayout GetSetLayout() const
        {
            return shadowLayout;
        }
        VkDescriptorSet GetFrameSet() const
        {
            return frameSet;
        }

        ShadowSettings settings;

      private:
        void computeCascades();
        void renderPass(PassExecution &passExec);

        DrawContext &drawContext;
        GPUSceneData &sceneData;

        AllocatedImage shadowAtlas;
        VkSampler shadowSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout shadowLayout = VK_NULL_HANDLE;
        VkDescriptorSet frameSet = VK_NULL_HANDLE;

        MaterialPipeline depthPipeline{};

        ShadowDataGPU shadowData{};
        glm::vec2 cascadeOffset[CASCADE_COUNT];
    };
} // namespace rgraph
