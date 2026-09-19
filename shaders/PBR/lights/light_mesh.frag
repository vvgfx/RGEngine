#version 450

#extension GL_GOOGLE_include_directive : require
#include "light_input_structures.glsl"
#include "../../PBR_helpers.glsl"

#define LIGHT_GRID_SET 3
#include "../../lighting/light_grid.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inPos;

layout(location = 0) out vec4 outFragColor;

void main()
{
    vec3 tempNormal = normalize(inNormal); // world space
    vec3 viewVec = (normalize(sceneData.cameraPos - inPos)).xyz;
    vec3 albedo = pow(inColor * texture(colorTex, vec2(inUV.s, inUV.t)).rgb, vec3(2.2)); // sRGB -> linear
    vec3 normal = tempNormal;
    vec2 metalRough = texture(metalRoughTex, inUV).bg;
    float metallic = metalRough.x * materialData.metal_rough_factors.x;
    float roughness = metalRough.y * materialData.metal_rough_factors.y;
    float ao = 1;

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0f);

    // Shared by both loops below; mirrors comp.frag.
    #define SHADE_LIGHT(lightVec, radiance)                                                                                    \
        {                                                                                                                      \
            float nDotL = max(dot(normal, lightVec), 0.0f);                                                                    \
            if (nDotL > 0.0)                                                                                                   \
            {                                                                                                                  \
                vec3 halfwayVec = normalize(viewVec + lightVec);                                                               \
                float NDF = DistributionGGX(normal, halfwayVec, roughness);                                                    \
                float G = GeometrySmith(normal, viewVec, lightVec, roughness);                                                 \
                vec3 F = FresnelSchlick(clamp(dot(halfwayVec, viewVec), 0.0f, 1.0f), F0);                                      \
                vec3 specular = (NDF * G * F) / (4.0 * max(dot(normal, viewVec), 0.0f) * nDotL + 0.001);                       \
                vec3 kD = (vec3(1.0f) - F) * (1.0 - metallic);                                                                 \
                Lo += (kD * albedo / PI + specular) * radiance * nDotL;                                                        \
            }                                                                                                                  \
        }

    // Directional lights reach everything, so they are never culled.
    for (int i = 0; i < lightData.numLights; i++)
    {
        if (lightData.pointLights[i].type != 0)
            continue;

        // glTF directional lights shine along the node's -Z, so the vector towards it is +Z.
        vec3 lightVec = normalize(lightData.pointLights[i].transform[2].xyz);
        vec3 radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity;

        SHADE_LIGHT(lightVec, radiance)
    }

    // Local lights come from this pixel's tile. Transparent fragments are in front of the opaque
    // depth the cull binned against, which is why that list keeps no near bound.
    // debugParams.z bypasses the grid, as in comp.frag.
    bool bypassCull = sceneData.debugParams.z > 0.5;
    uint tile = lightTileIndex(gl_FragCoord.xy);
    uint tileLights = bypassCull ? uint(lightData.numLights) : lightTileCount(tile);

    for (uint t = 0u; t < tileLights; t++)
    {
        int i = bypassCull ? int(t) : int(lightTileEntry(tile, t));
        if (bypassCull && lightData.pointLights[i].type == 0)
            continue;

        vec3 lightDistVec = lightData.pointLights[i].transform[3].xyz - inPos.xyz;

        // squared compare: rejects without a sqrt, and touches only one matrix column
        float distSq = dot(lightDistVec, lightDistVec);
        float range = lightData.pointLights[i].range;
        if (distSq > range * range)
            continue;

        vec3 lightVec = lightDistVec / sqrt(distSq);
        vec3 radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity / max(distSq, 1e-4);

        SHADE_LIGHT(lightVec, radiance)
    }

    vec3 ambient = vec3(0.03f) * albedo * ao;

    // linear HDR: the post pass owns tonemapping and gamma
    outFragColor = vec4(ambient + Lo, 1.0);
}
