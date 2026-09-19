#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    /**
     * @brief Bins point lights into 16x16 screen tiles so shading only touches lights that reach it.
     *
     * Without this every shaded pixel loops all 97 of Bistro's lights even though each reaches only
     * 2-8 units. The resulting grid is also what makes volumetric lamp cones affordable and lets
     * point-light shadows skip lights that touch no visible tile.
     */
    class LightCullFeature : public IFeature
    {
      public:
        static constexpr uint32_t TILE_SIZE = 16;
        static constexpr uint32_t MAX_LIGHTS_PER_TILE = 64;
        static constexpr uint32_t MAX_LIGHTS = 128;

        LightCullFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData, VkExtent3D screenExtent);

        /// No-op: the cull pass has to be declared between the geometry and composite passes, so
        /// DeferredRenderingFeature calls RegisterCullPass at that point instead.
        void Register(Rendergraph *builder) override
        {
        }

        void RegisterCullPass(Rendergraph *builder);

        /// Bound by the shading passes to read the grid.
        VkDescriptorSetLayout GetSetLayout() const
        {
            return readLayout;
        }
        VkDescriptorSet GetFrameSet() const
        {
            return readSet;
        }

        bool enabled = true;

      private:
        struct LightGPU
        {
            glm::mat4 transform;
            glm::vec3 color;
            float intensity;
            float range;
            int type;
            int shadowIndex;
            float _pad[1];
        };

        struct LightBlockGPU
        {
            LightGPU lights[MAX_LIGHTS];
            int numLights;
        };

        struct PushConstants
        {
            glm::uvec4 dims; // xy = tile counts, zw = screen size
        };

        void run(PassExecution &passExec);

        DrawContext &drawContext;
        GPUSceneData &sceneData;

        glm::uvec2 tileCount{};
        VkExtent3D screenExtent{};

        // Persistent and GPU-only: rewritten every frame, never read back.
        AllocatedBuffer gridBuffer;
        AllocatedBuffer indexBuffer;
        AllocatedBuffer infoBuffer;

        VkDescriptorSetLayout cullLayout = VK_NULL_HANDLE;
        VkPipelineLayout cullPipelineLayout = VK_NULL_HANDLE;
        VkPipeline cullPipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout readLayout = VK_NULL_HANDLE;
        VkDescriptorSet readSet = VK_NULL_HANDLE;

        VkSampler sampler = VK_NULL_HANDLE;
        DescriptorAllocatorGrowable descriptorAllocator;
    };
} // namespace rgraph
