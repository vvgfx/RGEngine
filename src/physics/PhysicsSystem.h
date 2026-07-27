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
    // physics depends on sgraph, never the reverse
    class PhysicsSystem
    {
      public:
        void init();

        // world transforms must be current; scale is baked into the collider
        void buildFromScene(const std::shared_ptr<sgraph::Scenegraph> &graph, const std::vector<sgraph::RigidBodySpec> &specs);

        void step(float frameDt);  // fixed-timestep accumulator, 1/60 with 4 substeps
        void sync();               // write dynamic/kinematic body poses back onto nodes
        void reset();              // re-drop: restore initial poses, zero velocities
        void cleanup();

        // Append world-space wireframe line segments for every collider proxy.
        void appendDebugLines(std::vector<DebugLineVertex> &out) const;

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

            // collider params in body-local space (scale already baked in) for debug draw
            glm::vec3 boxHalf{0.5f};
            glm::vec3 shapeCenter{0.f};
            float radius = 0.5f;
            glm::vec3 capA{0.f};
            glm::vec3 capB{0.f};
            std::shared_ptr<sgraph::Scene> geometry; // hull wireframe = source mesh edges

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
