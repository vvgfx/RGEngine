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
    class DDGIFeature;

    class DeferredRenderingFeature : public IFeature
    {
      public:
        DeferredRenderingFeature(DrawContext &drawContext, VkDevice _device, GPUSceneData &gpuSceneData, VkDescriptorSetLayout gpuSceneLayout,
                                 MaterialSystemCreateInfo &materialSystemCreateInfo, DeletionQueue &delQueue, DDGIFeature *ddgiFeature);

        void Register(Rendergraph *builder) override;

      private:
        // lighting data struct.
        struct PointLight
        {
            glm::mat4 transform;
            glm::vec3 color;
            float intensity;
            float range;
            int type;      // 0 = directional, 1 = spot, 2 = point
            float _pad[2]; // pad to 96 bytes
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

        // supplies set 3 of the composite pipeline (probe atlases + volume constants)
        DDGIFeature *ddgiFeature;
    };
} // namespace rgraph
