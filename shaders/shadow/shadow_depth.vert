#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

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

// lightViewProj is baked per cascade on the CPU, so this needs no descriptor set at all.
layout(push_constant) uniform constants
{
    mat4 lightViewProj;
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
}
PushConstants;

void main()
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = PushConstants.lightViewProj * PushConstants.modelMatrix * vec4(v.position, 1.0);
}
