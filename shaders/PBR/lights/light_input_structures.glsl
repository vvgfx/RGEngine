layout(set = 0, binding = 0) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection; // w for sun power
    vec4 sunlightColor;
    vec4 cameraPos;
    mat4 invViewproj;
}
sceneData;

struct PointLight
{
    mat4 transform;
    vec3 color;
    float intensity;
    float range;
    int type; // 0 = directional, 1 = spot, 2 = point
};

layout(set = 1, binding = 0) uniform LightData
{
    PointLight pointLights[128];
    int numLights;
}
lightData;

layout(set = 2, binding = 0) uniform GLTFMaterialData
{
    vec4 colorFactors;
    vec4 metal_rough_factors;
}
materialData;

layout(set = 2, binding = 1) uniform sampler2D colorTex;
layout(set = 2, binding = 2) uniform sampler2D metalRoughTex;
