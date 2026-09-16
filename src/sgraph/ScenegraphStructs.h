#pragma once

#include <memory>
struct DrawContext;
struct MeshAsset;
struct LightingData;

namespace sgraph
{
    // base class for a renderable dynamic object
    class INode
    {
      public:
        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) = 0;
    };

    // implementation of a drawable scene node.
    // the scene node can hold children and will also keep a transform to propagate
    // to them
    struct Node : public INode
    {

        // parent pointer must be a weak pointer to avoid circular dependencies
        std::weak_ptr<Node> parent;
        std::vector<std::shared_ptr<Node>> children;

        glm::mat4 localTransform;
        glm::mat4 worldTransform;

        void refreshTransform(const glm::mat4 &parentMatrix)
        {
            worldTransform = parentMatrix * localTransform;
            for (const auto &c : children)
            {
                c->refreshTransform(worldTransform);
            }
        }

        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
        {
            // draw children
            for (auto &c : children)
            {
                c->Draw(topMatrix, ctx);
            }
        }
    };

    struct MeshNode : public Node
    {

        std::shared_ptr<MeshAsset> mesh;

        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
    };

    struct LightNode : public Node
    {
        std::shared_ptr<LightingData> lightingData;

        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
    };

} // namespace sgraph