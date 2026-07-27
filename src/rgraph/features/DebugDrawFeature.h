#pragma once
#include "../IFeature.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <cstddef>
#include <vector>

namespace rgraph
{
    // LINE_LIST overlay over the composited scene; the physics system supplies world-space vertices each frame
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
