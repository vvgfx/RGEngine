layout(set = 0, binding = 0) uniform sampler2D inPosition;
layout(set = 0, binding = 1) uniform sampler2D inNormal;
layout(set = 0, binding = 2) uniform sampler2D inAlbedo;
layout(set = 0, binding = 3) uniform sampler2D inMetalllicRoughness;

layout(set = 1, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection; // w for sun power
    vec4 sunlightColor;
    vec4 cameraPos;
    mat4 invViewproj;
    vec4 ssaoParams;
    vec4 debugParams;
}
sceneData;

struct PointLight
{
    mat4 transform;
    vec3 color;
    float intensity;
    float range;
    int type;        // 0 = directional, 1 = spot, 2 = point
    int shadowIndex; // slot in the local shadow atlas, or -1
};

layout(set = 2, binding = 0) uniform LightData
{
    PointLight pointLights[128];
    int numLights;
}
lightData;
