#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

// Scene data at set 0, binding 0 (mirrors GPUSceneData / mrt_input_structures.glsl).
layout(set = 0, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
    vec4 cameraPos;
}
sceneData;

// Line vertices are pre-transformed into world space on the CPU; the shader only
// applies the camera. Matches the C++ DebugLineVertex { vec4 position; vec4 color; }.
struct LineVertex
{
    vec4 position;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer LineBuffer
{
    LineVertex vertices[];
};

layout(push_constant) uniform constants
{
    LineBuffer lineBuffer;
}
PushConstants;

layout(location = 0) out vec4 outColor;

void main()
{
    LineVertex v = PushConstants.lineBuffer.vertices[gl_VertexIndex];
    gl_Position = sceneData.viewproj * vec4(v.position.xyz, 1.0f);
    outColor = v.color;
}
