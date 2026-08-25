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
        void buildFromScene(const std::shared_ptr<sgraph::Scenegraph> &graph, const std::vector<sgraph::RigidBodySpec> &specs,
                            const std::vector<sgraph::MagnetSpec> &magnetSpecs);

        void step(float frameDt); // fixed-timestep accumulator, 1/60 with 4 substeps
        void stepOnce();          // exactly one substep, for stepping through while paused
        void sync();              // box3d -> scenegraph. Writes localTransform, so later rebakes agree with it.
        void reset();             // re-drop: restore initial poses, zero velocities
        void cleanup();

        void drawDebug(std::vector<DebugLineVertex> &out);

        // Magnet discs and pole axes, red north / blue south.
        void drawMagnets(std::vector<DebugLineVertex> &out) const;

        // one arrow per pole, along last substep's force
        void drawMagnetForces(std::vector<DebugLineVertex> &out) const;
        float largestPoleForce() const;

        bool magnetsEnabled = false;
        float magnetStrength = 0.5f; // pole strength, uncalibrated: the paper gives no force data
        float magnetArrowScale = 1.0e-3f;
        float magnetArrowMax = 2.0f; // I think we need this to prevent infinity lines.
        int magnetPairsLastStep = 0;

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
        void substep();
        void applyDebugForce();
        void applyMagnetForces();

        struct Magnet
        {
            glm::vec3 localPos{0.f};
            glm::vec3 axis{0.f, 1.f, 0.f};
            float polarity = 1.f;
            float thickness = 1.5f; // poles sit on the two faces, this far apart
        };

        // Magnets expanded to their poles in world space, rebuilt each substep.
        struct WorldPole
        {
            glm::vec3 pos;
            float charge;
            float radius;
            std::size_t body;
            glm::vec3 force{0.f};
        };
        std::vector<WorldPole> poles;

        struct Body
        {
            std::shared_ptr<sgraph::Node> node; // authored node this body drives
            std::string name;                   // so the UI can target a body
            b3BodyId id{};
            sgraph::RigidBodySpec::Body type = sgraph::RigidBodySpec::Body::Static;
            glm::vec3 bakedScale{1.f}; // scale baked into collider; re-applied to the visual on sync
            b3Pos initialPos{};
            b3Quat initialRot{};
            std::vector<Magnet> magnets;
        };

        b3WorldId world{};
        bool initialized = false;
        float accumulator = 0.f;
        std::vector<Body> bodies;
        std::vector<b3HullData *> ownedHulls; // heap hulls from b3CreateHull, freed in cleanup
    };
} // namespace physics
