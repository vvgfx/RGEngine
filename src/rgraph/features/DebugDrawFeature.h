#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <cstddef>
#include <vector>

namespace rgraph
{
    // Draws Box3D collider wireframes as a LINE_LIST overlay on top of the composited
    // scene, using the scene's view/projection. The line vertices are produced on the
    // CPU each frame (world-space) by the physics system and handed in via setLines().
    // Toggled with `enabled`.
    class DebugDrawFeature : public IFeature
    {
      public:
        DebugDrawFeature(VkDevice device, GPUSceneData &sceneData, VkDescriptorSetLayout gpuSceneLayout, VkFormat colorFormat, VkFormat depthFormat,
                         DeletionQueue &delQueue);

        void Register(Rendergraph *builder) override;

        void setLines(std::vector<DebugLineVertex> lines)
        {
            lineVerts = std::move(lines);
        }

        bool enabled = true;

      private:
        void draw(PassExecution &passExec);

        MaterialPipeline pipeline;
        VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
        GPUSceneData &sceneData;
        std::vector<DebugLineVertex> lineVerts;

        static constexpr std::size_t MAX_VERTS = 20000;
    };
} // namespace rgraph
