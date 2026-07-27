#include "PhysicsSystem.h"
#include "sgraph/Scenegraph.h"
#include "vk_loader.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fmt/core.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace physics;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
namespace
{
    b3BodyType mapType(sgraph::RigidBodySpec::Body b)
    {
        switch (b)
        {
        case sgraph::RigidBodySpec::Body::Dynamic:
            return b3_dynamicBody;
        case sgraph::RigidBodySpec::Body::Kinematic:
            return b3_kinematicBody;
        case sgraph::RigidBodySpec::Body::Static:
        default:
            return b3_staticBody;
        }
    }

    // TRS decompose (no shear; our authored transforms are pure translate/rotate/scale).
    void decompose(const glm::mat4 &m, glm::vec3 &t, glm::quat &r, glm::vec3 &s)
    {
        t = glm::vec3(m[3]);
        glm::vec3 c0(m[0]), c1(m[1]), c2(m[2]);
        s = {glm::length(c0), glm::length(c1), glm::length(c2)};
        glm::mat3 rot(c0 / (s.x != 0.f ? s.x : 1.f), c1 / (s.y != 0.f ? s.y : 1.f), c2 / (s.z != 0.f ? s.z : 1.f));
        r = glm::normalize(glm::quat_cast(rot));
    }

    // Union AABB of all surfaces in a glTF geometry (mesh-local space).
    void geometryBounds(const sgraph::Scene &sc, glm::vec3 &mn, glm::vec3 &mx)
    {
        mn = glm::vec3(FLT_MAX);
        mx = glm::vec3(-FLT_MAX);
        for (const auto &[name, mesh] : sc.meshes)
        {
            for (const GeoSurface &surf : mesh->surfaces)
            {
                mn = glm::min(mn, surf.bounds.origin - surf.bounds.extents);
                mx = glm::max(mx, surf.bounds.origin + surf.bounds.extents);
            }
        }
        if (mn.x > mx.x) // no surfaces -> degenerate; fall back to unit box
        {
            mn = glm::vec3(-0.5f);
            mx = glm::vec3(0.5f);
        }
    }

    b3Pos toPos(const glm::vec3 &v)
    {
        b3Pos p;
        p.x = v.x;
        p.y = v.y;
        p.z = v.z;
        return p;
    }

    b3Quat toQuat(const glm::quat &q)
    {
        b3Quat r;
        r.v = b3Vec3{q.x, q.y, q.z};
        r.s = q.w;
        return r;
    }

    // ---- debug-line generation ----
    void addLine(std::vector<DebugLineVertex> &out, const glm::mat4 &M, const glm::vec3 &a, const glm::vec3 &b, const glm::vec4 &c)
    {
        glm::vec3 wa = glm::vec3(M * glm::vec4(a, 1.f));
        glm::vec3 wb = glm::vec3(M * glm::vec4(b, 1.f));
        out.push_back({glm::vec4(wa, 1.f), c});
        out.push_back({glm::vec4(wb, 1.f), c});
    }

    // Sample an arc: p(theta) = center + radius*(cos*e1 + sin*e2), theta in [t0,t1].
    void arc(std::vector<DebugLineVertex> &out, const glm::mat4 &M, const glm::vec3 &center, const glm::vec3 &e1, const glm::vec3 &e2,
             float radius, float t0, float t1, int segs, const glm::vec4 &c)
    {
        glm::vec3 prev(0.f);
        for (int i = 0; i <= segs; i++)
        {
            float th = t0 + (t1 - t0) * (float)i / (float)segs;
            glm::vec3 p = center + radius * (std::cos(th) * e1 + std::sin(th) * e2);
            if (i > 0)
                addLine(out, M, prev, p, c);
            prev = p;
        }
    }

    constexpr float TWO_PI = 6.2831853f;
    constexpr float PI = 3.14159265f;
} // namespace

// ---------------------------------------------------------------------------
void PhysicsSystem::init()
{
    b3WorldDef def = b3DefaultWorldDef(); // default gravity {0,-10,0}
    world = b3CreateWorld(&def);
    initialized = true;
}

void PhysicsSystem::buildFromScene(const std::shared_ptr<sgraph::Scenegraph> &graph, const std::vector<sgraph::RigidBodySpec> &specs)
{
    if (!initialized)
        init();

    for (const sgraph::RigidBodySpec &spec : specs)
    {
        auto opt = graph->getNode(spec.nodeName);
        if (!opt.has_value())
        {
            fmt::print("physics: rigidbody references unknown node '{}'\n", spec.nodeName);
            continue;
        }
        auto node = std::dynamic_pointer_cast<sgraph::Node>(opt.value());
        auto leaf = std::dynamic_pointer_cast<sgraph::GLTFLeafNode>(opt.value());
        if (!node || !leaf || !leaf->geometry)
        {
            fmt::print("physics: node '{}' has no glTF geometry to derive a collider from\n", spec.nodeName);
            continue;
        }

        // pose from the node's accumulated world transform; bake scale into geometry
        glm::vec3 t, s;
        glm::quat r;
        decompose(node->worldTransform, t, r, s);

        Body body;
        body.node = node;
        body.type = spec.body;
        body.shape = spec.shape;
        body.bakedScale = s;
        body.geometry = leaf->geometry;

        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = mapType(spec.body);
        bd.position = toPos(t);
        bd.rotation = toQuat(r);
        body.id = b3CreateBody(world, &bd);
        body.initialPos = bd.position;
        body.initialRot = bd.rotation;

        b3ShapeDef sd = b3DefaultShapeDef();
        sd.density = spec.density;

        glm::vec3 mn, mx;
        geometryBounds(*leaf->geometry, mn, mx);
        glm::vec3 center = 0.5f * (mn + mx) * s; // scaled, body-local
        glm::vec3 half = 0.5f * (mx - mn) * s;   // scaled half-extents

        switch (spec.shape)
        {
        case sgraph::RigidBodySpec::Shape::Box:
        {
            b3BoxHull bh = b3MakeOffsetBoxHull(half.x, half.y, half.z, b3Vec3{center.x, center.y, center.z});
            b3CreateHullShape(body.id, &sd, &bh.base);
            body.boxHalf = half;
            body.shapeCenter = center;
            break;
        }
        case sgraph::RigidBodySpec::Shape::Sphere:
        {
            float rad = std::max({half.x, half.y, half.z});
            b3Sphere sph;
            sph.center = b3Vec3{center.x, center.y, center.z};
            sph.radius = rad;
            b3CreateSphereShape(body.id, &sd, &sph);
            body.radius = rad;
            body.shapeCenter = center;
            break;
        }
        case sgraph::RigidBodySpec::Shape::Capsule:
        {
            // dominant axis = largest scaled half-extent; radius = larger of the other two
            int axis = 0;
            if (half.y > half[axis])
                axis = 1;
            if (half.z > half[axis])
                axis = 2;
            float rad = 0.f;
            for (int i = 0; i < 3; i++)
                if (i != axis)
                    rad = std::max(rad, half[i]);
            glm::vec3 dir(0.f);
            dir[axis] = 1.f;
            float seg = std::max(half[axis] - rad, 0.f); // half-distance between hemisphere centers
            glm::vec3 a = center + dir * seg;
            glm::vec3 b = center - dir * seg;
            b3Capsule cap;
            cap.center1 = b3Vec3{a.x, a.y, a.z};
            cap.center2 = b3Vec3{b.x, b.y, b.z};
            cap.radius = rad;
            b3CreateCapsuleShape(body.id, &sd, &cap);
            body.capA = a;
            body.capB = b;
            body.radius = rad;
            break;
        }
        case sgraph::RigidBodySpec::Shape::Hull:
        {
            std::vector<b3Vec3> pts;
            for (const auto &[name, mesh] : leaf->geometry->meshes)
                for (const glm::vec3 &p : mesh->positions)
                    pts.push_back(b3Vec3{p.x * s.x, p.y * s.y, p.z * s.z});
            if (pts.empty())
            {
                fmt::print("physics: hull node '{}' has no vertices\n", spec.nodeName);
                break;
            }
            b3HullData *h = b3CreateHull(pts.data(), (int)pts.size(), (int)pts.size());
            ownedHulls.push_back(h);
            b3CreateHullShape(body.id, &sd, h);
            break;
        }
        }

        bodies.push_back(std::move(body));
    }

    fmt::print("physics: built {} bodies\n", bodies.size());
}

void PhysicsSystem::step(float frameDt)
{
    if (!initialized)
        return;
    accumulator = std::min(accumulator + frameDt, 0.25f); // clamp: anti spiral-of-death
    const float fixedDt = 1.0f / 60.0f;
    while (accumulator >= fixedDt)
    {
        b3World_Step(world, fixedDt, 4);
        accumulator -= fixedDt;
    }
}

void PhysicsSystem::sync()
{
    for (Body &b : bodies)
    {
        if (b.type == sgraph::RigidBodySpec::Body::Static)
            continue;
        b3Pos p = b3Body_GetPosition(b.id);
        b3Quat q = b3Body_GetRotation(b.id);
        glm::vec3 pos((float)p.x, (float)p.y, (float)p.z);
        glm::quat rot(q.s, q.v.x, q.v.y, q.v.z); // glm order (w,x,y,z)
        // re-apply the baked scale so the visual mesh keeps its authored size
        glm::mat4 m = glm::translate(glm::mat4(1.f), pos) * glm::toMat4(rot) * glm::scale(glm::mat4(1.f), b.bakedScale);
        if (b.node)
            b.node->worldTransform = m;
    }
}

void PhysicsSystem::reset()
{
    for (Body &b : bodies)
    {
        if (b.type == sgraph::RigidBodySpec::Body::Static)
            continue;
        b3Body_SetTransform(b.id, b.initialPos, b.initialRot);
        b3Body_SetLinearVelocity(b.id, b3Vec3{0.f, 0.f, 0.f});
        b3Body_SetAngularVelocity(b.id, b3Vec3{0.f, 0.f, 0.f});
        b3Body_SetAwake(b.id, true);
    }
    accumulator = 0.f;
}

void PhysicsSystem::cleanup()
{
    for (b3HullData *h : ownedHulls)
        b3DestroyHull(h);
    ownedHulls.clear();
    bodies.clear();
    if (initialized)
    {
        b3DestroyWorld(world);
        initialized = false;
    }
}

void PhysicsSystem::appendDebugLines(std::vector<DebugLineVertex> &out) const
{
    const glm::vec4 kBox{1.0f, 0.85f, 0.1f, 1.f};
    const glm::vec4 kSphere{0.2f, 0.9f, 1.0f, 1.f};
    const glm::vec4 kCapsule{1.0f, 0.3f, 0.9f, 1.f};
    const glm::vec4 kHull{0.4f, 1.0f, 0.4f, 1.f};

    for (const Body &b : bodies)
    {
        // current world pose (no scale: collider params already include baked scale)
        b3Pos p = b3Body_GetPosition(b.id);
        b3Quat q = b3Body_GetRotation(b.id);
        glm::vec3 pos((float)p.x, (float)p.y, (float)p.z);
        glm::quat rot(q.s, q.v.x, q.v.y, q.v.z);
        glm::mat4 M = glm::translate(glm::mat4(1.f), pos) * glm::toMat4(rot);

        switch (b.shape)
        {
        case sgraph::RigidBodySpec::Shape::Box:
        {
            glm::vec3 c = b.shapeCenter, h = b.boxHalf;
            glm::vec3 crn[8];
            for (int i = 0; i < 8; i++)
                crn[i] = c + glm::vec3((i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z);
            const int e[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7}, {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
            for (auto &ed : e)
                addLine(out, M, crn[ed[0]], crn[ed[1]], kBox);
            break;
        }
        case sgraph::RigidBodySpec::Shape::Sphere:
        {
            glm::vec3 c = b.shapeCenter;
            glm::vec3 X{1, 0, 0}, Y{0, 1, 0}, Z{0, 0, 1};
            arc(out, M, c, X, Y, b.radius, 0.f, TWO_PI, 24, kSphere);
            arc(out, M, c, Y, Z, b.radius, 0.f, TWO_PI, 24, kSphere);
            arc(out, M, c, X, Z, b.radius, 0.f, TWO_PI, 24, kSphere);
            break;
        }
        case sgraph::RigidBodySpec::Shape::Capsule:
        {
            glm::vec3 axis = b.capA - b.capB;
            float len = glm::length(axis);
            axis = len > 1e-5f ? axis / len : glm::vec3(0, 1, 0);
            glm::vec3 ref = std::abs(axis.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
            glm::vec3 u = glm::normalize(glm::cross(axis, ref));
            glm::vec3 v = glm::cross(axis, u);
            float r = b.radius;
            arc(out, M, b.capA, u, v, r, 0.f, TWO_PI, 20, kCapsule); // ring at cap A
            arc(out, M, b.capB, u, v, r, 0.f, TWO_PI, 20, kCapsule); // ring at cap B
            for (const glm::vec3 &d : {u, -u, v, -v})
                addLine(out, M, b.capA + r * d, b.capB + r * d, kCapsule); // connectors
            arc(out, M, b.capA, u, axis, r, 0.f, PI, 12, kCapsule);       // cap A domes
            arc(out, M, b.capA, v, axis, r, 0.f, PI, 12, kCapsule);
            arc(out, M, b.capB, u, -axis, r, 0.f, PI, 12, kCapsule); // cap B domes
            arc(out, M, b.capB, v, -axis, r, 0.f, PI, 12, kCapsule);
            break;
        }
        case sgraph::RigidBodySpec::Shape::Hull:
        {
            if (!b.geometry)
                break;
            glm::vec3 sc = b.bakedScale;
            for (const auto &[name, mesh] : b.geometry->meshes)
            {
                const auto &pos3 = mesh->positions;
                const auto &idx = mesh->indices;
                for (std::size_t i = 0; i + 2 < idx.size(); i += 3)
                {
                    glm::vec3 a = pos3[idx[i + 0]] * sc;
                    glm::vec3 bb = pos3[idx[i + 1]] * sc;
                    glm::vec3 cc = pos3[idx[i + 2]] * sc;
                    addLine(out, M, a, bb, kHull);
                    addLine(out, M, bb, cc, kHull);
                    addLine(out, M, cc, a, kHull);
                }
            }
            break;
        }
        }
    }
}
