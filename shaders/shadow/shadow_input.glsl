#ifndef SHADOW_INPUT_GLSL
#define SHADOW_INPUT_GLSL

const int SHADOW_CASCADE_COUNT = 4;

layout(set = SHADOW_SET, binding = 0) uniform ShadowData
{
    mat4 cascadeViewProj[SHADOW_CASCADE_COUNT];
    vec4 cascadeSplits; // view-space far distance per cascade
    vec4 params;        // x = texel world size, y = pcf radius, z = atlas dimension, w = enabled
}
shadowData;

layout(set = SHADOW_SET, binding = 1) uniform sampler2DShadow shadowAtlas;

/// Cascades are packed 2x2 in one atlas; this maps a cascade-local UV into it.
vec2 cascadeAtlasUV(vec2 uv, int cascade)
{
    vec2 offset = vec2(float(cascade & 1), float(cascade >> 1)) * 0.5;
    return uv * 0.5 + offset;
}

float sampleShadow(vec3 worldPos, float viewDepth, vec3 N, vec3 L)
{
    if (shadowData.params.w < 0.5)
    {
        return 1.0;
    }

    int cascade = SHADOW_CASCADE_COUNT - 1;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++)
    {
        if (viewDepth < shadowData.cascadeSplits[i])
        {
            cascade = i;
            break;
        }
    }

    // Normal-offset bias: pushing along the normal scales with texel size, which handles slope
    // without the acne/peter-panning tradeoff a constant depth bias forces.
    float texelWorld = shadowData.params.x * exp2(float(cascade));
    float slope = clamp(1.0 - dot(N, L), 0.0, 1.0);
    vec3 offsetPos = worldPos + N * texelWorld * (1.0 + 2.0 * slope) * 1.5;

    vec4 lightSpace = shadowData.cascadeViewProj[cascade] * vec4(offsetPos, 1.0);
    lightSpace.xyz /= lightSpace.w;

    vec2 uv = lightSpace.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || lightSpace.z < 0.0 || lightSpace.z > 1.0)
    {
        return 1.0;
    }

    // 3x3 PCF. sampler2DShadow gives hardware comparison, so each tap is already filtered.
    float texel = 1.0 / shadowData.params.z;
    float radius = shadowData.params.y;
    float sum = 0.0;

    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 tapUV = cascadeAtlasUV(uv + vec2(x, y) * texel * radius, cascade);
            sum += texture(shadowAtlas, vec3(tapUV, lightSpace.z));
        }
    }

    return sum / 9.0;
}

#endif
