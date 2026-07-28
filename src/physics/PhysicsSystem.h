#pragma once

#include "sgraph/ScenegraphStructs.h"
#include "vk_types.h"
#include <box3d/box3d.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace sgraph
{
    class Scenegraph;
}

namespace physics
{
    // Owns the Box3D world and the authored-node -> rigid-body mapping.
    // One-directional: physics depends on sgraph, never the reverse.
    class PhysicsSystem
    {
      public:
        void init();

        // Graph world transforms must be current (call root->refreshTransform first):
        // each body's initial pose is its node's decomposed world transform, with
        // scale baked into the collider (Box3D bodies have no scale).
        void buildFromScene(const std::shared_ptr<sgraph::Scenegraph> &graph, const std::vector<sgraph::RigidBodySpec> &specs);

        void step(float frameDt); // fixed-timestep accumulator, 1/60 with 4 substeps
        void sync();              // write dynamic/kinematic body poses back onto nodes
        void reset();             // re-drop: restore initial poses, zero velocities
        void cleanup();

        // Fill `out` with world-space collider wireframes via b3World_Draw callbacks.
        void drawDebug(std::vector<DebugLineVertex> &out);

        std::size_t bodyCount() const
        {
            return bodies.size();
        }

      private:
        struct Body
        {
            std::shared_ptr<sgraph::Node> node; // authored node this body drives
            b3BodyId id{};
            sgraph::RigidBodySpec::Body type = sgraph::RigidBodySpec::Body::Static;
            sgraph::RigidBodySpec::Shape shape = sgraph::RigidBodySpec::Shape::Box;
            glm::vec3 bakedScale{1.f}; // scale baked into collider; re-applied to the visual on sync
            b3Pos initialPos{};
            b3Quat initialRot{};
        };

        b3WorldId world{};
        bool initialized = false;
        float accumulator = 0.f;
        std::vector<Body> bodies;
        std::vector<b3HullData *> ownedHulls; // heap hulls from b3CreateHull, freed in cleanup
    };
} // namespace physics
