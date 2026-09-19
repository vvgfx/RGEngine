#pragma once
#include "../IFeature.h"
#include "AccelStructure.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <glm/glm.hpp>

namespace rgraph
{
    /// Runtime-tweakable DDGI parameters. The graph rebuilds every frame, so changes apply next frame.
    struct DDGISettings
    {
        bool enabled = true;
        bool shadowRays = true;
        bool showProbes = false;
        bool sunShadows = true;
        int aoRays = 2;        // per pixel in the composite pass; 0 disables
        float aoRadius = 0.0f; // world units; 0 means "derive from probe spacing"
        float probeRadius = 0.05f; // fraction of probe spacing; purely a debug-view size
        int raysPerProbe = 128;
        float hysteresis = 0.97f;
        float normalBias = 0.25f;
        float viewBias = 0.8f;
        float depthSharpness = 50.0f;
        float irradianceGamma = 5.0f;
        glm::vec3 skyColor{0.1f, 0.2f, 0.4f};
    };

    /**
     * @brief Dynamic diffuse GI from a grid of ray-traced irradiance probes.
     *
     * Probe atlases are owned here rather than registered with the rendergraph: Build() resets tracked
     * layouts to UNDEFINED every frame, which would discard the temporal accumulation this depends on.
     * They stay in VK_IMAGE_LAYOUT_GENERAL for their whole lifetime and are synchronised by hand.
     */
    class DDGIFeature : public IFeature
    {
      public:
        DDGIFeature(VulkanEngine *engine, VkDevice device, const AccelStructure &accel, const DrawContext &sceneSnapshot,
                    DrawContext &drawContext, GPUSceneData &sceneData, VkDescriptorSetLayout sceneDataLayout, VkFormat colorFormat,
                    VkFormat depthFormat, DeletionQueue &delQueue);

        void Register(Rendergraph *builder) override;

        /// Declared by DDGIDebugFeature so the probe spheres draw after the composite pass.
        void RegisterDebug(Rendergraph *builder);

        VkDescriptorSetLayout GetSetLayout() const
        {
            return ddgiLayout;
        }

        /// Valid only after this frame's trace pass has run; the composite pass binds it at set 3.
        VkDescriptorSet GetFrameSet() const
        {
            return frameSet;
        }

        DDGISettings settings;

      private:
        // GPU mirror of DDGIVolumeBlock in ddgi_common.glsl. vec4 rows keep std140 layout trivial.
        struct VolumeGPU
        {
            glm::vec4 origin;
            glm::vec4 spacing;
            glm::ivec4 counts;
            glm::vec4 blend;
            glm::vec4 misc;
            glm::vec4 skyColor;
            glm::vec4 flags; // x = ray-traced shadows in the composite pass
        };

        struct PushConstants
        {
            glm::mat4 rayRotation;
            glm::vec4 flags;
        };

        struct LightGPU
        {
            glm::mat4 transform;
            glm::vec3 color;
            float intensity;
            float range;
            int type;
            float _pad[2];
        };

        static constexpr int MAX_LIGHTS = 128;

        struct LightBlockGPU
        {
            LightGPU lights[MAX_LIGHTS];
            int numLights;
        };

        bool active() const;
        void fitVolumeToScene(const DrawContext &snapshot);
        void createImages(VkDevice device, DeletionQueue &delQueue);
        void createPipelines(VkDevice device, DeletionQueue &delQueue);

        void tracePass(PassExecution &passExec);
        void debugProbePass(PassExecution &passExec);
        void blendPass(PassExecution &passExec, VkPipeline pipeline, int tileSize);

        /// The atlases are untracked, so the graph emits no barriers for them.
        static void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst);

        VulkanEngine *engine;
        const AccelStructure &accel;
        DrawContext &drawContext;
        GPUSceneData &sceneData;
        VkFormat colorFormat;
        VkFormat depthFormat;

        glm::vec3 volumeOrigin{0.0f};
        glm::vec3 volumeSpacing{1.0f};
        glm::ivec3 probeCounts{32, 12, 32};
        float maxRayDistance = 100.0f;

        AllocatedImage probeRayData;
        AllocatedImage probeIrradiance;
        AllocatedImage probeDistance;
        VkSampler probeSampler;

        VkDescriptorSetLayout ddgiLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout sceneLayout = VK_NULL_HANDLE;
        VkPipelineLayout tracePipelineLayout = VK_NULL_HANDLE;
        VkPipelineLayout blendPipelineLayout = VK_NULL_HANDLE;

        VkPipeline tracePipeline = VK_NULL_HANDLE;
        VkPipeline blendIrradiancePipeline = VK_NULL_HANDLE;
        VkPipeline blendDistancePipeline = VK_NULL_HANDLE;
        VkPipelineLayout debugPipelineLayout = VK_NULL_HANDLE;
        VkPipeline debugPipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout sceneDataLayout = VK_NULL_HANDLE;

        VkDescriptorSet frameSet = VK_NULL_HANDLE;
        PushConstants pushConstants{};
        uint32_t frameIndex = 0;

        int numProbes() const
        {
            return probeCounts.x * probeCounts.y * probeCounts.z;
        }
    };

    /// Thin adapter so the probe overlay can be registered after the deferred feature; execution
    /// order in the graph is simply the order features were added.
    class DDGIDebugFeature : public IFeature
    {
      public:
        explicit DDGIDebugFeature(DDGIFeature *ddgi) : ddgi(ddgi)
        {
        }

        void Register(Rendergraph *builder) override
        {
            ddgi->RegisterDebug(builder);
        }

      private:
        DDGIFeature *ddgi;
    };
} // namespace rgraph
