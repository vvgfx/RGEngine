#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"

namespace rgraph
{
    struct LightDebugSettings
    {
        bool enabled = false;
        bool showDirectional = false; // the sun has no meaningful radius; off by default
        float radiusScale = 1.0f;     // scale the drawn sphere without touching the light's range
    };

    /**
     * @brief Draws each punctual light as a wireframe sphere at its range.
     *
     * Diagnostic only. Lighting bugs where a lamp ends up inside a wall or at the wrong scale are
     * invisible in the shaded image -- you see a glow and cannot tell whether the light or the
     * geometry moved. Three great circles per light make position and reach directly readable.
     */
    class LightDebugFeature : public IFeature
    {
      public:
        LightDebugFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData, VkFormat colorFormat,
                          VkFormat depthFormat);

        void Register(Rendergraph *builder) override;

        LightDebugSettings settings;

      private:
        /// Fits in the guaranteed 128-byte push constant range, so no buffer or descriptor set.
        struct SpherePush
        {
            glm::mat4 viewproj;
            glm::vec4 posRadius;
            glm::vec4 color;
        };

        void renderPass(PassExecution &passExec);

        DrawContext &drawContext;
        GPUSceneData &sceneData;

        MaterialPipeline pipeline{};
    };
} // namespace rgraph
