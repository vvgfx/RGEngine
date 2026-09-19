#ifndef LOCAL_SHADOW_GLSL
#define LOCAL_SHADOW_GLSL

const int LOCAL_SHADOW_MAX_FACES = 96;

layout(set = LOCAL_SHADOW_SET, binding = 0) uniform LocalShadowData
{
    mat4 faceViewProj[LOCAL_SHADOW_MAX_FACES];
    vec4 tileRect[LOCAL_SHADOW_MAX_FACES]; // xy = atlas UV origin, zw = size
    vec4 params;                           // x = pcf radius, y = atlas dim, z = normal bias, w = enabled
}
localShadow;

layout(set = LOCAL_SHADOW_SET, binding = 1) uniform sampler2DShadow localShadowAtlas;

/// Cube face for a direction. Must match the order LocalShadowFeature builds its matrices in.
int localShadowFace(vec3 d)
{
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z)
    {
        return d.x > 0.0 ? 0 : 1;
    }
    if (a.y >= a.z)
    {
        return d.y > 0.0 ? 2 : 3;
    }
    return d.z > 0.0 ? 4 : 5;
}

float sampleLocalShadow(int shadowIndex, vec3 worldPos, vec3 lightPos, vec3 N)
{
    if (shadowIndex < 0 || localShadow.params.w < 0.5)
    {
        return 1.0;
    }

    vec3 toFrag = worldPos - lightPos;
    int face = shadowIndex * 6 + localShadowFace(toFrag);

    // Normal offset rather than a depth bias: it scales with surface slope and avoids trading acne
    // for peter-panning on these small-radius lamps.
    vec3 biased = worldPos + N * localShadow.params.z;

    vec4 clip = localShadow.faceViewProj[face] * vec4(biased, 1.0);
    if (clip.w <= 0.0)
    {
        return 1.0;
    }

    vec3 ndc = clip.xyz / clip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    vec4 rect = localShadow.tileRect[face];
    float texel = 1.0 / localShadow.params.y;
    float radius = localShadow.params.x;

    float sum = 0.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            // clamp inside the tile so a PCF tap cannot bleed into a neighbouring light's face
            vec2 tileUV = clamp(uv + vec2(x, y) * texel * radius / rect.z, vec2(0.001), vec2(0.999));
            sum += texture(localShadowAtlas, vec3(rect.xy + tileUV * rect.zw, ndc.z));
        }
    }

    return sum / 9.0;
}

#endif
