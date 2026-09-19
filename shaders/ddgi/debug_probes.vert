#version 450

#extension GL_GOOGLE_include_directive : require

#define DDGI_SET 0
#include "ddgi_common.glsl"

layout(set = 1, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
    vec4 cameraPos;
    mat4 invViewproj;
}
sceneData;

layout(push_constant) uniform DebugPushConstants
{
    vec4 params; // x = probe radius
}
pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) flat out int outProbeIndex;

// A UV sphere generated on the fly, so no probe mesh has to be uploaded.
const int SLICES = 8;
const int STACKS = 8;

void main()
{
    const ivec2 corners[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));

    int quad = gl_VertexIndex / 6;
    ivec2 off = corners[gl_VertexIndex % 6];

    float u = float(quad % SLICES + off.x) / float(SLICES);
    float v = float(quad / SLICES + off.y) / float(STACKS);

    float phi = u * 2.0 * DDGI_PI;
    float theta = v * DDGI_PI;

    vec3 n = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    vec3 centre = ddgiProbePosition(ddgiProbeCoord(gl_InstanceIndex));

    outNormal = n;
    outProbeIndex = gl_InstanceIndex;

    gl_Position = sceneData.viewproj * vec4(centre + n * pc.params.x, 1.0);
}
