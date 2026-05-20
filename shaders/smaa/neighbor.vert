#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : enable

#define SMAA_GLSL_4 1
#define SMAA_FLIP_Y 0
#define SMAA_PRESET_HIGH 1
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0

layout(push_constant) uniform PC { vec4 rtMetrics; } pc;
#define SMAA_RT_METRICS pc.rtMetrics

#include "SMAA.h"

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec4 offset;

void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    texcoord = uv;
    SMAANeighborhoodBlendingVS(texcoord, offset);
}
