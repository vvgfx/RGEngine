#pragma once

#include "physics/PhysicsSystem.h"
#include "rgraph/Rendergraph.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DebugDrawFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include "rgraph/features/SkyboxFeature.h"
#include "sgraph/Scenegraph.h"
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

    // --- physics PoC: authored scenegraph drives Box3D, debug overlay inspects it ---
    std::shared_ptr<sgraph::Scenegraph> scenegraph;
    physics::PhysicsSystem physics;
    std::shared_ptr<rgraph::DebugDrawFeature> debugFeature;

    std::chrono::steady_clock::time_point lastPhysicsTime;
    bool physicsPaused = false;
    bool showDebugDraw = true;

    // --- sky + directional sun ---
    std::shared_ptr<rgraph::SkyboxFeature> skyboxFeature;
    glm::vec3 sunDirection{-0.35f, -1.0f, -0.25f}; // from sun toward scene
    glm::vec3 sunColor{1.0f, 0.96f, 0.9f};
    float sunIntensity = 3.5f;
    float sunDiskSize = 0.02f;
    glm::vec3 skyHorizon{0.72f, 0.80f, 0.92f};
    glm::vec3 skyZenith{0.20f, 0.42f, 0.82f};
    glm::vec3 skyGround{0.28f, 0.26f, 0.24f};
    int skyMode = 1; // 0 procedural, 1 texture (your equirect sky is the default)

    // --- directional shadow map ---
    bool shadowsEnabled = true;
    float shadowBias = 0.0025f;
};