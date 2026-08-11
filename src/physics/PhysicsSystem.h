#pragma once

#include "sgraph/ScenegraphStructs.h"
#include "vk_types.h"
#include <box3d/box3d.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
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

        void step(float frameDt); // fixed-timestep accumulator, 1/60 with 4 substeps
        void sync();              // box3d -> scenegraph. Writes localTransform, so later rebakes agree with it.
        void reset();             // re-drop: restore initial poses, zero velocities
        void cleanup();

        void drawDebug(std::vector<DebugLineVertex> &out);

        std::size_t bodyCount() const
        {
            return bodies.size();
        }

        std::string debugTarget;
        glm::vec3 debugForceWeights{-2.f, 0.f, 0.f}; // world axes, in multiples of the body's weight
        bool debugForceActive = false;

        std::vector<std::string> bodyNames() const;
        float bodyMass(const std::string &name) const;

      private:
        void applyDebugForce();

        struct Body
        {
            std::shared_ptr<sgraph::Node> node; // authored node this body drives
            std::string name;                   // so the UI can target a body
            b3BodyId id{};
            sgraph::RigidBodySpec::Body type = sgraph::RigidBodySpec::Body::Static;
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
