#pragma once
#include "../IFeature.h"
#include "MaterialSystem.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{

    /**
     * @brief An implementation of IFeature that does deferred rendering
     *
     */
    class ShadowFeature;
    class LocalShadowFeature;
    class LightCullFeature;

    class DeferredRenderingFeature : public IFeature
    {
      public:
        DeferredRenderingFeature(DrawContext &drawContext, VkDevice _device, GPUSceneData &gpuSceneData, VkDescriptorSetLayout gpuSceneLayout,
                                 MaterialSystemCreateInfo &materialSystemCreateInfo, DeletionQueue &delQueue, ShadowFeature *shadowFeature,
                                 LocalShadowFeature *localShadowFeature, LightCullFeature *lightCullFeature);

        void Register(Rendergraph *builder) override;

      private:
        // lighting data struct.
        struct PointLight
        {
            glm::mat4 transform;
            glm::vec3 color;
            float intensity;
            float range;
            int type;        // 0 = directional, 1 = spot, 2 = point
            int shadowIndex; // slot in the local shadow atlas, or -1
            float _pad[1];   // pad to 96 bytes
        };

        static constexpr int MAX_LIGHTS = 128;

        struct LightData
        {
            PointLight pointLights[MAX_LIGHTS];
            int numLights;
        };

        void createPipelines(MaterialSystemCreateInfo &materialSystemCreateInfo);

        void createImages(DeletionQueue &delQueue);
        // execution lambdas for run.
        void geometryPass(PassExecution &passExec);

        void compositePass(PassExecution &passExec);

        void transparentPass(PassExecution &passExec);

        // images
        AllocatedImage position_gbuf;
        AllocatedImage normal_gbuf;
        AllocatedImage albedo_gbuf;
        AllocatedImage metalrough_gbuf;
        AllocatedImage depth_gbuf;

        VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
        VkDescriptorSetLayout lightDescriptorSetLayout;
        VkDescriptorSetLayout compDescriptorSetLayout;

        MaterialPipeline geometryPipeline;
        MaterialPipeline compositePipeline;
        MaterialPipeline transparentPipeline;

        DrawContext &drawContext;
        GPUSceneData &gpuSceneData;

        VkSampler defaultSampler;

        // supplies set 3 of the composite pipeline (cascade matrices + shadow atlas)
        ShadowFeature *shadowFeature;

        // supplies set 4 (point-light cube shadow atlas)
        LocalShadowFeature *localShadowFeature;

        // supplies set 5 (per-tile light lists)
        LightCullFeature *lightCullFeature;
    };
} // namespace rgraph
