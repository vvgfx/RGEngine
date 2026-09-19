#version 450

#extension GL_GOOGLE_include_directive : require

#define DDGI_SET 0
#include "../PBR_helpers.glsl"
#include "ddgi_common.glsl"
#include "ddgi_sample.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) flat in int inProbeIndex;

layout(location = 0) out vec4 outFragColor;

void main()
{
    vec2 irrSize = vec2(textureSize(ddgiIrradianceTex, 0));

    // Show this probe's own octahedral irradiance, unfiltered by the cage weights, so a bad probe is
    // visible on its own rather than averaged away.
    vec3 irr = texture(ddgiIrradianceTex, ddgiProbeUV(inProbeIndex, normalize(inNormal), DDGI_IRR_TILE, irrSize)).rgb;
    irr = pow(irr, vec3(DDGI_GAMMA * 0.5));

    outFragColor = vec4(irr, 1.0);
}
