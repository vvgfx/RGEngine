#pragma once
#include "../IFeature.h"
#include "vk_engine.h"

namespace rgraph
{

    class SkyboxFeature : public IFeature
    {
      public:
        SkyboxFeature(VkDevice _device, DeletionQueue &delQueue);

        void Register(Rendergraph *rgraphInstance) override;


      private:

          VkPipeline pipeline;
          VkPipelineLayout pipelineLayout;
          VkDescriptorSetLayout descriptorLayout;
          VkDescriptorSet descriptorSet;
          // DescriptorAllocator descriptorAllocator; // don't know if I need this yet.
    };
} // namespace rgraph