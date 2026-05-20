#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : enable

#define SMAA_GLSL_4 1
#define SMAA_FLIP_Y 0
#define SMAA_PRESET_HIGH 1

layout(push_constant) uniform PC { vec4 rtMetrics; } pc;
#define SMAA_RT_METRICS pc.rtMetrics

#include "SMAA.h"

layout(set = 0, binding = 0) uniform sampler2D colorTex;
layout(set = 0, binding = 1) uniform sampler2D blendTex;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = SMAANeighborhoodBlendingPS(texcoord, offset, colorTex, blendTex);
}
