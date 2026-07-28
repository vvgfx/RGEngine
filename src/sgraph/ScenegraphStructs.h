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

        glm::mat4 localTransform{1.f};
        glm::mat4 worldTransform{1.f};

        virtual ~Node() = default;

        void refreshTransform(const glm::mat4 &parentMatrix)
        {
            worldTransform = parentMatrix * localTransform;
            for (auto c : children)
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

    // ---- Authored scenegraph node types (physics PoC) ------------------------
    // The authored graph is authoritative; glTF is a pure geometry source referenced
    // by a GLTFLeafNode. Transforms accumulate via refreshTransform (world = parent * local).

    struct Scene; // full definition in vk_loader.h (glTF geometry container)

    // A pure grouping node (identity transform).
    struct GroupNode : public Node
    {
    };

    // A transform node whose local matrix is composed from translate/rotate/scale
    // commands (each post-multiplied on, so authored order is TRS as written).
    struct TransformNode : public Node
    {
        void applyTranslate(const glm::vec3 &t);
        void applyRotate(float degrees, const glm::vec3 &axis);
        void applyScale(const glm::vec3 &s);
    };

    // A leaf node referencing an external glTF used purely as geometry.
    struct GLTFLeafNode : public Node
    {
        std::string gltfName;
        std::shared_ptr<Scene> geometry;

        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
    };

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