#pragma once

#include "rgraph/Rendergraph.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include <box3d/box3d.h>
#include <chrono>
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

    // AllocatedImages for MSAA. TODO: Move these out later.
    AllocatedImage msaaColor;
    AllocatedImage msaaDepth;

    void createMsaaImages();

    // --- box3d physics demo: a cube falling onto a static ground slab ---
    b3WorldId physicsWorld;
    b3BodyId fallingBox;
    std::chrono::steady_clock::time_point lastPhysicsTime;
    float physicsAccumulator = 0.f;
    bool physicsPaused = false;
};