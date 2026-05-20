#pragma once

#include "rgraph/Rendergraph.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include "rgraph/features/SMAAFeature.h"
#include <memory>
#include <vk_descriptors.h>
#include <vk_engine.h>
#include <vk_types.h>

class RGEngine : public VulkanEngine
{
  public:
    void init() override;

  protected:
    // functions
    void init_pipelines() override;

    void init_default_data() override;

    void cleanupOnChildren() override;

    void update_scene() override;

    void draw() override;

    void imGuiAddParams() override;

    // gltf data
    std::unordered_map<std::string, std::shared_ptr<sgraph::Scene>> loadedScenes;

    rgraph::Rendergraph builder;
    std::shared_ptr<rgraph::ComputeBackgroundFeature> computeFeature;
    std::shared_ptr<rgraph::PBRShadingFeature> PBRFeature;
    std::shared_ptr<rgraph::DeferredRenderingFeature> deferredFeature;
    std::shared_ptr<rgraph::SMAAFeature> smaaFeature;

    // AllocatedImages for MSAA. TODO: Move these out later.
    AllocatedImage msaaColor;
    AllocatedImage msaaDepth;

    // Final antialiased output, blitted to swapchain.
    AllocatedImage smaaOutput;

    void testRendergraph();

    void createMsaaImages();
    void createSmaaOutput();
};