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
    vec4 ssaoParams;
    vec4 debugParams;
}
sceneData;

layout(set = 1, binding = 0) uniform GLTFMaterialData
{

    vec4 colorFactors;
    vec4 metal_rough_factors;
    vec4 extra0; // x = alpha cutoff (0 = off), y = 1 when metalRoughTex is a spec/gloss map
    vec4 extra1; // xyz = emissive colour (can exceed 1)
}
materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
layout(set = 1, binding = 3) uniform sampler2D normalTex;
layout(set = 1, binding = 4) uniform sampler2D emissiveTex;