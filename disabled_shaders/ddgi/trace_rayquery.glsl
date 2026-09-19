#ifndef DDGI_TRACE_RAYQUERY_GLSL
#define DDGI_TRACE_RAYQUERY_GLSL

#include "ddgi_hit.glsl"

// Matches Vertex in vk_types.h (48 bytes).
struct RTVertex
{
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexRef
{
    RTVertex vertices[];
};

layout(buffer_reference, std430) readonly buffer IndexRef
{
    uint indices[];
};

// Mirrors GPUGeometryInfo in AccelStructure.h; indexed by the TLAS instance custom index.
struct GeometryInfo
{
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4 colorFactor;
    uint firstIndex;
    uint albedoTexIndex;
    uint pad0;
    uint pad1;
};

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1, std430) readonly buffer GeometryBuffer
{
    GeometryInfo geometries[];
};

layout(set = 2, binding = 0) uniform sampler2D bindlessTextures[];

/// Any-hit occlusion test, terminating on the first intersection.
bool TraceShadow(vec3 origin, vec3 dir, float tMax)
{
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.01, dir, tMax);
    while (rayQueryProceedEXT(rq))
    {
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

bool TraceScene(vec3 origin, vec3 dir, float tMax, out DDGIHit hit)
{
    hit.hit = false;
    hit.backface = false;
    hit.t = tMax;
    hit.position = origin + dir * tMax;
    hit.normal = -dir;
    hit.albedo = vec3(0.0);

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001, dir, tMax);
    while (rayQueryProceedEXT(rq))
    {
    }

    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
    {
        return false;
    }

    int instIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    int primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    float t = rayQueryGetIntersectionTEXT(rq, true);
    bool frontFace = rayQueryGetIntersectionFrontFaceEXT(rq, true);
    mat4x3 objToWorld = rayQueryGetIntersectionObjectToWorldEXT(rq, true);

    GeometryInfo geo = geometries[instIdx];
    VertexRef vb = VertexRef(geo.vertexAddress);
    IndexRef ib = IndexRef(geo.indexAddress);

    // A ray query reports a triangle ordinal, not vertices: resolving it needs the index buffer.
    uint base = geo.firstIndex + 3u * uint(primIdx);
    RTVertex v0 = vb.vertices[ib.indices[base + 0u]];
    RTVertex v1 = vb.vertices[ib.indices[base + 1u]];
    RTVertex v2 = vb.vertices[ib.indices[base + 2u]];

    vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    vec3 localNormal = normalize(v0.normal * b.x + v1.normal * b.y + v2.normal * b.z);
    vec3 worldNormal = normalize(mat3(objToWorld) * localNormal);

    vec2 uv = vec2(v0.uv_x, v0.uv_y) * b.x + vec2(v1.uv_x, v1.uv_y) * b.y + vec2(v2.uv_x, v2.uv_y) * b.z;

    vec3 albedo = texture(bindlessTextures[nonuniformEXT(geo.albedoTexIndex)], uv).rgb * geo.colorFactor.rgb;
    // match the G-buffer, which linearises with pow(2.2) in mrt.frag
    albedo = pow(albedo, vec3(2.2));

    hit.hit = true;
    hit.backface = !frontFace;
    hit.t = t;
    hit.position = origin + dir * t;
    hit.normal = frontFace ? worldNormal : -worldNormal;
    hit.albedo = albedo;
    return true;
}

#endif
