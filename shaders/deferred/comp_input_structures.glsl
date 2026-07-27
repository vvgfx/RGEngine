layout(set = 0, binding = 0) uniform sampler2D inPosition;
layout(set = 0, binding = 1) uniform sampler2D inNormal;
layout(set = 0, binding = 2) uniform sampler2D inAlbedo;
layout(set = 0, binding = 3) uniform sampler2D inMetalllicRoughness;
layout(set = 0, binding = 4) uniform sampler2D shadowMap;

layout(set = 1, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection; // w for sun power
    vec4 sunlightColor;
    vec4 cameraPos;
    mat4 sunViewProj;  // world -> sun light clip space
    vec4 shadowParams; // x = bias, y = enabled
}
sceneData;

struct PointLight
{
    mat4 transform;
    vec3 color;
    float intensity;
    float range;
};

layout(set = 2, binding = 0) uniform LightData
{
    PointLight pointLights[25];
    int numLights;
}
lightData;
