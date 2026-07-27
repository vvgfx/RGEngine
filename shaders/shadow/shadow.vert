#version 450

#extension GL_EXT_buffer_reference : require

// the shadow vertex shader reads sunViewProj from set 0
layout(set = 0, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
    vec4 cameraPos;
    mat4 sunViewProj;
    vec4 shadowParams;
}
sceneData;

struct Vertex
{
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{
    Vertex vertices[];
};

layout(push_constant) uniform constants
{
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
}
PushConstants;

void main()
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = sceneData.sunViewProj * PushConstants.modelMatrix * vec4(v.position, 1.0f);
}
