#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{

class SMAAFeature : public IFeature
{
  public:
    SMAAFeature(VkDevice device, VkExtent3D extent, DeletionQueue &delQueue);
    void Register(Rendergraph *builder) override;

  private:
    void createImages(DeletionQueue &delQueue);
    void createPipelines();
    void createLUTs(DeletionQueue &delQueue);

    void edgePass(PassExecution &passExec);
    void blendWeightPass(PassExecution &passExec);
    void neighborBlendPass(PassExecution &passExec);

    // Intermediate images, registered with the rendergraph for layout tracking
    AllocatedImage smaaEdges;
    AllocatedImage smaaBlend;

    // Precomputed SMAA lookup tables, uploaded once at startup
    AllocatedImage areaTexture;
    AllocatedImage searchTexture;

    VkDescriptorSetLayout edgeDescLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout blendDescLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout neighborDescLayout = VK_NULL_HANDLE;

    MaterialPipeline edgePipeline;
    MaterialPipeline blendPipeline;
    MaterialPipeline neighborPipeline;

    VkSampler linearSampler = VK_NULL_HANDLE;

    VkDevice _device;
    VkExtent3D _extent;
};

} // namespace rgraph
