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
    class DeferredRenderingFeature : public IFeature
    {
      public:
        DeferredRenderingFeature(DrawContext &drawContext, VkDevice _device, GPUSceneData &gpuSceneData, VkDescriptorSetLayout gpuSceneLayout,
                                 MaterialSystemCreateInfo &materialSystemCreateInfo, DeletionQueue &delQueue);

        void Register(Rendergraph *builder) override;

      private:
        // lighting data struct.
        struct PointLight
        {
            glm::mat4 transform;
            glm::vec3 color;
            float intensity;
            float range;
            float _pad[3]; // pad to 96 bytes
        };

        struct LightData
        {
            PointLight pointLights[25];
            int numLights;
        };

        void createPipelines(MaterialSystemCreateInfo &materialSystemCreateInfo);

        // execution lambdas for run.
        void geometryPass(PassExecution &passExec);

        void compositePass(PassExecution &passExec);

        void transparentPass(PassExecution &passExec);

        VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
        VkDescriptorSetLayout lightDescriptorSetLayout;
        VkDescriptorSetLayout compDescriptorSetLayout;

        MaterialPipeline geometryPipeline;
        MaterialPipeline compositePipeline;
        MaterialPipeline transparentPipeline;

        DrawContext &drawContext;
        GPUSceneData &gpuSceneData;

        VkSampler defaultSampler;
    };
} // namespace rgraph
