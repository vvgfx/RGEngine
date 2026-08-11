#include "PhysicsSystem.h"
#include "sgraph/Scenegraph.h"
#include "vk_loader.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <fmt/core.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace physics;

// helpers + Box3D debug-draw callbacks
namespace
{
    constexpr float TWO_PI = 6.2831853f;
    constexpr float PI = 3.14159265f;
    constexpr float GRAVITY = 10.0f; // magnitude of b3DefaultWorldDef's {0,-10,0}

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

    // no shear: authored transforms are pure TRS
    void decompose(const glm::mat4 &m, glm::vec3 &t, glm::quat &r, glm::vec3 &s)
    {
        t = glm::vec3(m[3]);
        glm::vec3 c0(m[0]), c1(m[1]), c2(m[2]);
        s = {glm::length(c0), glm::length(c1), glm::length(c2)};
        glm::mat3 rot(c0 / (s.x != 0.f ? s.x : 1.f), c1 / (s.y != 0.f ? s.y : 1.f), c2 / (s.z != 0.f ? s.z : 1.f));
        r = glm::normalize(glm::quat_cast(rot));
    }

    // union AABB of all surfaces, mesh-local
    void geometryBounds(const sgraph::Scene &sc, glm::vec3 &mn, glm::vec3 &mx)
    {
        mn = glm::vec3(FLT_MAX);
        mx = glm::vec3(-FLT_MAX);
        for (const auto &entry : sc.meshes)
        {
            for (const GeoSurface &surf : entry.second->surfaces)
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

    glm::vec3 toGlm(const b3Vec3 &v)
    {
        return glm::vec3(v.x, v.y, v.z);
    }

    // ---- Box3D debug-draw path ----
    // endpoint pairs in shape-local space, built once and transformed to world each frame
    struct DebugShapeProxy
    {
        std::vector<glm::vec3> segments;
    };

    void pushSeg(std::vector<glm::vec3> &s, const glm::vec3 &a, const glm::vec3 &b)
    {
        s.push_back(a);
        s.push_back(b);
    }

    // arc as local segments: p(theta) = c + r*(cos*e1 + sin*e2), theta in [t0,t1]
    void arcLocal(std::vector<glm::vec3> &s, const glm::vec3 &c, const glm::vec3 &e1, const glm::vec3 &e2, float r, float t0, float t1, int segs)
    {
        glm::vec3 prev(0.f);
        for (int i = 0; i <= segs; i++)
        {
            float th = t0 + (t1 - t0) * (float)i / (float)segs;
            glm::vec3 p = c + r * (std::cos(th) * e1 + std::sin(th) * e2);
            if (i > 0)
            {
                pushSeg(s, prev, p);
            }
            prev = p;
        }
    }

    // we own the returned handle
    void *box3dCreateDebugShape(const b3DebugShape *ds, void * /*userContext*/)
    {
        auto *proxy = new DebugShapeProxy();

        switch (ds->type)
        {
        case b3_hullShape: // boxes are hulls too -> real Box3D hull edges
        {
            const b3HullData *hull = ds->hull;
            const b3Vec3 *pts = b3GetHullPoints(hull);
            const b3HullHalfEdge *edges = b3GetHullEdges(hull);
            if (pts && edges)
            {
                // edgeCount is the half-edge count; draw each undirected edge once (i < twin)
                for (int i = 0; i < hull->edgeCount; i++)
                {
                    int twin = edges[i].twin;
                    if (i < twin)
                    {
                        pushSeg(proxy->segments, toGlm(pts[edges[i].origin]), toGlm(pts[edges[twin].origin]));
                    }
                }
            }
            break;
        }
        case b3_sphereShape:
        {
            const b3Sphere *sp = ds->sphere;
            glm::vec3 c = toGlm(sp->center);
            float r = sp->radius;
            arcLocal(proxy->segments, c, {1, 0, 0}, {0, 1, 0}, r, 0.f, TWO_PI, 24);
            arcLocal(proxy->segments, c, {0, 1, 0}, {0, 0, 1}, r, 0.f, TWO_PI, 24);
            arcLocal(proxy->segments, c, {1, 0, 0}, {0, 0, 1}, r, 0.f, TWO_PI, 24);
            break;
        }
        case b3_capsuleShape:
        {
            const b3Capsule *cap = ds->capsule;
            glm::vec3 a = toGlm(cap->center1), b = toGlm(cap->center2);
            float r = cap->radius;
            glm::vec3 axis = a - b;
            float len = glm::length(axis);
            axis = len > 1e-5f ? axis / len : glm::vec3(0, 1, 0);
            glm::vec3 ref = std::abs(axis.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
            glm::vec3 u = glm::normalize(glm::cross(axis, ref));
            glm::vec3 v = glm::cross(axis, u);
            arcLocal(proxy->segments, a, u, v, r, 0.f, TWO_PI, 20); // ring at cap A
            arcLocal(proxy->segments, b, u, v, r, 0.f, TWO_PI, 20); // ring at cap B
            for (const glm::vec3 &d : {u, -u, v, -v})
            {
                pushSeg(proxy->segments, a + r * d, b + r * d); // connectors
            }
            arcLocal(proxy->segments, a, u, axis, r, 0.f, PI, 12); // cap A domes
            arcLocal(proxy->segments, a, v, axis, r, 0.f, PI, 12);
            arcLocal(proxy->segments, b, u, -axis, r, 0.f, PI, 12); // cap B domes
            arcLocal(proxy->segments, b, v, -axis, r, 0.f, PI, 12);
            break;
        }
        default:
            break; // compound/mesh/heightfield not used in this PoC
        }

        return proxy;
    }

    void box3dDestroyDebugShape(void *userShape, void * /*userContext*/)
    {
        delete static_cast<DebugShapeProxy *>(userShape);
    }

    glm::vec4 hexToColor(b3HexColor c)
    {
        unsigned v = (unsigned)c & 0x00FFFFFFu; // low 24 bits are RGB; high byte is a material preset
        return glm::vec4(((v >> 16) & 0xFF) / 255.f, ((v >> 8) & 0xFF) / 255.f, (v & 0xFF) / 255.f, 1.f);
    }

    // returning true keeps box3d drawing (b3DebugDraw contract)
    bool box3dDrawShape(void *userShape, b3WorldTransform transform, b3HexColor color, void *context)
    {
        auto *out = static_cast<std::vector<DebugLineVertex> *>(context);
        auto *proxy = static_cast<DebugShapeProxy *>(userShape);
        if (!out || !proxy)
        {
            return true;
        }

        glm::vec3 p((float)transform.p.x, (float)transform.p.y, (float)transform.p.z);
        glm::quat q(transform.q.s, transform.q.v.x, transform.q.v.y, transform.q.v.z);
        glm::mat4 M = glm::translate(glm::mat4(1.f), p) * glm::toMat4(q);
        glm::vec4 col = hexToColor(color);

        for (std::size_t i = 0; i + 1 < proxy->segments.size(); i += 2)
        {
            glm::vec3 a = glm::vec3(M * glm::vec4(proxy->segments[i], 1.f));
            glm::vec3 b = glm::vec3(M * glm::vec4(proxy->segments[i + 1], 1.f));
            out->push_back({glm::vec4(a, 1.f), col});
            out->push_back({glm::vec4(b, 1.f), col});
        }
        return true;
    }
} // namespace

void PhysicsSystem::init()
{
    b3WorldDef def = b3DefaultWorldDef(); // default gravity {0,-10,0}
    // so b3World_Draw can render collider wireframes
    def.createDebugShape = &box3dCreateDebugShape;
    def.destroyDebugShape = &box3dDestroyDebugShape;
    world = b3CreateWorld(&def);
    initialized = true;
}

void PhysicsSystem::buildFromScene(const std::shared_ptr<sgraph::Scenegraph> &graph, const std::vector<sgraph::RigidBodySpec> &specs)
{
    if (!initialized)
    {
        init();
    }

    for (const sgraph::RigidBodySpec &spec : specs)
    {
        auto opt = graph->getNode(spec.nodeName);
        if (!opt.has_value())
        {
            fmt::print("physics: rigidbody references unknown node '{}'\n", spec.nodeName);
            continue;
        }
        auto scene = std::dynamic_pointer_cast<sgraph::Scene>(opt.value());
        if (!scene)
        {
            fmt::print("physics: node '{}' has no glTF geometry to derive a collider from\n", spec.nodeName);
            continue;
        }

        // box3d bodies have no scale, so bake it into the geometry
        glm::vec3 t, s;
        glm::quat r;
        decompose(scene->worldTransform, t, r, s);

        Body body;
        body.node = scene;
        body.name = spec.nodeName;
        body.type = spec.body;
        body.bakedScale = s;

        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = mapType(spec.body);
        bd.position = toPos(t);
        bd.rotation = toQuat(r);
        body.id = b3CreateBody(world, &bd);
        body.initialPos = bd.position;
        body.initialRot = bd.rotation;

        // a sleeping body ignores applied force
        if (spec.body != sgraph::RigidBodySpec::Body::Static)
        {
            b3Body_EnableSleep(body.id, false);
        }

        b3ShapeDef sd = b3DefaultShapeDef();
        if (spec.body != sgraph::RigidBodySpec::Body::Static)
        {
            sd.density = spec.density; // static bodies have no mass to derive from it
        }

        glm::vec3 center{0.f}, half{0.5f};
        if (spec.shape != sgraph::RigidBodySpec::Shape::Hull)
        {
            glm::vec3 mn, mx;
            geometryBounds(*scene, mn, mx);
            center = 0.5f * (mn + mx) * s; // scaled, body-local
            half = 0.5f * (mx - mn) * s;   // scaled half-extents
        }

        switch (spec.shape)
        {
        case sgraph::RigidBodySpec::Shape::Box:
        {
            b3BoxHull bh = b3MakeOffsetBoxHull(half.x, half.y, half.z, b3Vec3{center.x, center.y, center.z});
            b3CreateHullShape(body.id, &sd, &bh.base);
            break;
        }
        case sgraph::RigidBodySpec::Shape::Sphere:
        {
            b3Sphere sph;
            sph.center = b3Vec3{center.x, center.y, center.z};
            sph.radius = std::max({half.x, half.y, half.z});
            b3CreateSphereShape(body.id, &sd, &sph);
            break;
        }
        case sgraph::RigidBodySpec::Shape::Capsule:
        {
            int axis = 0;
            if (half.y > half[axis])
            {
                axis = 1;
            }
            if (half.z > half[axis])
            {
                axis = 2;
            }
            float rad = 0.f;
            for (int i = 0; i < 3; i++)
            {
                if (i != axis)
                {
                    rad = std::max(rad, half[i]);
                }
            }
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
            break;
        }
        case sgraph::RigidBodySpec::Shape::Hull:
        {
            // one hull per mesh: box3d hulls are convex, so concave bodies need a decomposed export
            constexpr int maxHullVerts = 40; // box3d's 255 half-edge cap puts V under ~44
            int hullCount = 0;
            for (const auto &entry : scene->meshes)
            {
                std::vector<b3Vec3> pts;
                pts.reserve(entry.second->positions.size());
                for (const glm::vec3 &p : entry.second->positions)
                {
                    pts.push_back(b3Vec3{p.x * s.x, p.y * s.y, p.z * s.z});
                }
                if (pts.empty())
                {
                    continue;
                }

                b3HullData *h = b3CreateHull(pts.data(), (int)pts.size(), maxHullVerts);
                if (h == nullptr) // box3d logs the reason; never hand a null to b3CreateHullShape
                {
                    fmt::print("physics: hull build failed for a mesh on '{}'\n", spec.nodeName);
                    continue;
                }
                ownedHulls.push_back(h);
                b3CreateHullShape(body.id, &sd, h);
                hullCount++;
            }
            if (hullCount == 0)
            {
                fmt::print("physics: hull node '{}' produced no collider\n", spec.nodeName);
            }
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
    {
        return;
    }
    accumulator = std::min(accumulator + frameDt, 0.25f); // clamp: anti spiral-of-death
    const float fixedDt = 1.0f / 60.0f;
    while (accumulator >= fixedDt)
    {
        applyDebugForce(); // inside the loop: b3World_Step zeroes the force it just consumed
        b3World_Step(world, fixedDt, 4);
        accumulator -= fixedDt;
    }
}

void PhysicsSystem::applyDebugForce()
{
    if (!debugForceActive || debugTarget.empty())
    {
        return;
    }
    for (Body &b : bodies)
    {
        if (b.name != debugTarget || b.type == sgraph::RigidBodySpec::Body::Static)
        {
            continue;
        }
        float weight = b3Body_GetMass(b.id) * GRAVITY;
        b3Body_ApplyForceToCenter(
            b.id, b3Vec3{debugForceWeights.x * weight, debugForceWeights.y * weight, debugForceWeights.z * weight}, true);
    }
}

std::vector<std::string> PhysicsSystem::bodyNames() const
{
    std::vector<std::string> names;
    names.reserve(bodies.size());
    for (const Body &b : bodies)
    {
        names.push_back(b.name);
    }
    return names;
}

float PhysicsSystem::bodyMass(const std::string &name) const
{
    for (const Body &b : bodies)
    {
        if (b.name == name)
        {
            return b3Body_GetMass(b.id);
        }
    }
    return 0.f;
}

void PhysicsSystem::sync()
{
    for (Body &b : bodies)
    {
        if (b.type == sgraph::RigidBodySpec::Body::Static)
        {
            continue;
        }
        b3Pos p = b3Body_GetPosition(b.id);
        b3Quat q = b3Body_GetRotation(b.id);
        glm::vec3 pos((float)p.x, (float)p.y, (float)p.z);
        glm::quat rot(q.s, q.v.x, q.v.y, q.v.z); // glm order (w,x,y,z)
        // re-apply the baked scale, box3d dropped it
        glm::mat4 m = glm::translate(glm::mat4(1.f), pos) * glm::toMat4(rot) * glm::scale(glm::mat4(1.f), b.bakedScale);
        if (!b.node)
        {
            continue;
        }

        glm::mat4 parentWorld{1.f};
        if (auto parentNode = b.node->parent.lock())
        {
            parentWorld = parentNode->worldTransform;
        }

        // box3d is world-space, localTransform isn't. Not cached: parents can move.
        b.node->localTransform = glm::inverse(parentWorld) * m;
        b.node->refreshTransform(parentWorld);
    }
}

void PhysicsSystem::reset()
{
    for (Body &b : bodies)
    {
        if (b.type == sgraph::RigidBodySpec::Body::Static)
        {
            continue;
        }
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
    {
        b3DestroyHull(h);
    }
    ownedHulls.clear();
    bodies.clear();
    if (initialized)
    {
        b3DestroyWorld(world); // also invokes destroyDebugShape for any built debug shapes
        initialized = false;
    }
}

void PhysicsSystem::drawDebug(std::vector<DebugLineVertex> &out)
{
    if (!initialized)
    {
        return;
    }
    b3DebugDraw draw = b3DefaultDebugDraw();
    draw.drawShapes = true;
    draw.DrawShapeFcn = &box3dDrawShape;
    draw.context = &out;
    // ensure the broadphase query covers the whole scene
    draw.drawingBounds = b3AABB{b3Vec3{-1e9f, -1e9f, -1e9f}, b3Vec3{1e9f, 1e9f, 1e9f}};
    b3World_Draw(world, &draw, UINT64_MAX);
}
