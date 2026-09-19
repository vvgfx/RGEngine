#pragma once

#include "rgraph/Rendergraph.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/ShadowFeature.h"
#include "rgraph/features/PostProcessFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
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

    rgraph::Rendergraph rgraphInstance;
    std::shared_ptr<rgraph::ComputeBackgroundFeature> computeFeature;
    std::shared_ptr<rgraph::PBRShadingFeature> PBRFeature;
    std::shared_ptr<rgraph::DeferredRenderingFeature> deferredFeature;
    std::shared_ptr<rgraph::ShadowFeature> shadowFeature;

    // mirrored into GPUSceneData::ssaoParams.x each frame; int so ImGui can edit it
    int ssaoSampleCount = 16;
    std::shared_ptr<rgraph::PostProcessFeature> postFeature;

    // AllocatedImages for MSAA. TODO: Move these out later.
    AllocatedImage msaaColor;
    AllocatedImage msaaDepth;

    // tonemapped, gamma-encoded, anti-aliased result; this is what reaches the swapchain
    AllocatedImage postImage;

    void createMsaaImages();
};