#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    struct LocalShadowSettings
    {
        bool enabled = false; // parked: re-enable once sun and lamp intensity are separable
        int maxLights = 12;      // shadow casters; each costs six cube faces
        float normalBias = 0.05f;
        float pcfRadius = 1.0f;
    };

    /**
     * @brief Cube shadow maps for point lights, packed into one atlas.
     *
     * All 96 of Bistro's lights casting is not affordable at six faces each, so the closest and
     * brightest few get tiles and the rest stay unshadowed. Their ranges are 2-8 units, so a missing
     * shadow on a distant fill light is hard to notice.
     */
    class LocalShadowFeature : public IFeature
    {
      public:
        static constexpr uint32_t ATLAS_DIM = 4096;
        static constexpr uint32_t TILE_RES = 256;          // plenty for a lamp with an 8 unit reach
        static constexpr uint32_t TILES_PER_ROW = ATLAS_DIM / TILE_RES;
        static constexpr uint32_t MAX_LIGHTS = 16;
        static constexpr uint32_t MAX_FACES = MAX_LIGHTS * 6;

        LocalShadowFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData);

        void Register(Rendergraph *builder) override;

        VkDescriptorSetLayout GetSetLayout() const
        {
            return setLayout;
        }
        VkDescriptorSet GetFrameSet() const
        {
            return frameSet;
        }

        /// Shadow slot for the light at `lightIndex` in DrawContext::lights, or -1 if it has none.
        int ShadowIndexFor(int lightIndex) const;

        LocalShadowSettings settings;

      private:
        /// Matches LocalShadowData in local_shadow.glsl.
        struct ShadowDataGPU
        {
            glm::mat4 faceViewProj[MAX_FACES];
            glm::vec4 tileRect[MAX_FACES]; // xy = atlas UV origin, zw = size
            glm::vec4 params;              // x = pcf radius, y = atlas dim, z = normal bias, w = enabled
        };

        struct Caster
        {
            int lightIndex;
            float importance;
        };

        void selectLights();
        void renderPass(PassExecution &passExec);

        DrawContext &drawContext;
        GPUSceneData &sceneData;

        AllocatedImage atlas;
        VkSampler shadowSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkDescriptorSet frameSet = VK_NULL_HANDLE;
        MaterialPipeline depthPipeline{};

        ShadowDataGPU shadowData{};
        std::vector<Caster> casters;      // selected this frame, most important first
        std::vector<int> shadowIndexByLight; // -1 when the light has no tile
    };
} // namespace rgraph
