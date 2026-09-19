#version 450

#extension GL_GOOGLE_include_directive : require
#include "mrt_input_structures.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inPosition;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMetalllicRoughness;

/**
 * Tangent frame from screen-space derivatives (Schueler's cotangent frame).
 *
 * The asset does carry TANGENT, but consuming it would mean widening Vertex and updating every
 * GLSL mirror including the ray-query hit shader, which would shift the BLAS vertex stride. This
 * is confined to one function and is visually equivalent on geometry this dense.
 */
mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv)
{
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // scale-invariant: guards against degenerate UVs producing a collapsed frame
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-12);
    return mat3(T * invmax, B * invmax, N);
}

void main()
{
    vec4 baseColor = texture(colorTex, inUV);

    // Alpha-masked foliage and window frames. Cutoff of 0 means the material is opaque.
    float cutoff = materialData.extra0.x;
    if (cutoff > 0.0 && baseColor.a < cutoff)
    {
        discard;
    }

    outPosition = vec4(inPosition, 1.0f);

    vec3 geometricNormal = normalize(inNormal);
    vec3 tangentNormal = texture(normalTex, inUV).xyz * 2.0 - 1.0;

    // A flat default decodes to +Z, so this is a no-op for materials without a normal map.
    outNormal = vec4(normalize(cotangentFrame(geometricNormal, inPosition, inUV) * tangentNormal), 1.0f);

    vec3 textureColor = inColor * baseColor.xyz;
    // convert to linear space.
    textureColor = pow(textureColor, vec3(2.2));

    // Emissive is packed into the three unused G-buffer channels rather than adding a fifth
    // attachment: it never participates in lighting, so it only needs to survive to the composite.
    vec3 emissive = texture(emissiveTex, inUV).rgb * materialData.extra1.rgb;

    outAlbedo = vec4(textureColor, emissive.r);

    float metallic;
    float roughness;

    if (materialData.extra0.y > 0.5)
    {
        // spec/gloss: RGB is specular colour, A is glossiness
        vec4 specGloss = texture(metalRoughTex, inUV);
        metallic = 0.0;
        roughness = 1.0 - specGloss.a * (1.0 - materialData.metal_rough_factors.y);
    }
    else
    {
        vec2 metalRough = texture(metalRoughTex, inUV).bg;
        metallic = metalRough.x * materialData.metal_rough_factors.x;
        roughness = metalRough.y * materialData.metal_rough_factors.y;
    }

    outMetalllicRoughness.x = metallic;
    outMetalllicRoughness.y = clamp(roughness, 0.03f, 1.0f);
    outMetalllicRoughness.z = emissive.g;
    outMetalllicRoughness.w = emissive.b;
}
