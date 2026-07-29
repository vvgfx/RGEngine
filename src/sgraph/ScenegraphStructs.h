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

        // localTransform is the pose. worldTransform is only ever parentWorld * localTransform, cached so nothing
        // walks the ancestor chain - read that one everywhere downstream.
        glm::mat4 localTransform{1.f};
        glm::mat4 worldTransform{1.f};

        virtual ~Node() = default;

        void refreshTransform(const glm::mat4 &parentMatrix)
        {
            worldTransform = parentMatrix * localTransform;
            for (auto &c : children)
            {
                c->refreshTransform(worldTransform);
            }
        }

        // Compose onto localTransform. Each post-multiplies, so authored order is TRS as written.
        void applyTranslate(const glm::vec3 &t);
        void applyRotate(float degrees, const glm::vec3 &axis);
        void applyScale(const glm::vec3 &s);

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

    // ---- Authored scenegraph node types (physics PoC) ------------------------
    // The authored graph is authoritative; a glTF file enters it as a Scene node (vk_loader.h).
    // Only nodes carrying extra data subclass Node - grouping and transforming are what a plain Node already does.

    // Parsed 'rigidbody' command. Neutral authoring data (no Box3D types) so the
    // physics layer depends on sgraph, never the reverse.
    struct RigidBodySpec
    {
        enum class Body
        {
            Static,
            Dynamic,
            Kinematic
        };
        enum class Shape
        {
            Box,
            Sphere,
            Capsule,
            Hull
        };

        std::string nodeName;
        Body body = Body::Static;
        Shape shape = Shape::Box;
        float density = 1.0f;
    };

} // namespace sgraph