#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <glm/glm.hpp>

namespace rgraph
{
    // background pixels only, after the deferred composite; geometry is masked out by the G-buffer normal
    class SkyboxFeature : public IFeature
    {
      public:
        // Must match shaders/sky/skybox.frag SkyParams (std140).
        struct Params
        {
            glm::mat4 invViewProj{1.f};
            glm::vec4 cameraPos{0.f};
            glm::vec4 sunDirection{0.f, -1.f, 0.f, 3.f}; // xyz dir, w intensity
            glm::vec4 sunColor{1.f, 0.96f, 0.9f, 0.02f}; // rgb, w disk size
            glm::vec4 horizon{0.70f, 0.80f, 0.95f, 1.f};
            glm::vec4 zenith{0.25f, 0.45f, 0.85f, 1.f};
            glm::vec4 ground{0.30f, 0.28f, 0.25f, 1.f};
            glm::ivec4 mode{0, 0, 0, 0}; // x: 0 procedural, 1 texture
        };

        SkyboxFeature(VkDevice device, VkFormat colorFormat, VkFormat depthFormat, DeletionQueue &delQueue);

        void Register(Rendergraph *builder) override;

        void setParams(const Params &p)
        {
            params = p;
        }
        // equirectangular; white fallback
        void setSkyTexture(AllocatedImage tex)
        {
            skyTexture = tex;
        }

        bool enabled = true;

      private:
        void draw(PassExecution &passExec);

        MaterialPipeline pipeline;
        VkDescriptorSetLayout skyUboLayout;
        VkDescriptorSetLayout inputLayout;
        VkSampler sampler;
        AllocatedImage skyTexture{};
        Params params;
    };
} // namespace rgraph
