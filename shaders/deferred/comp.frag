#version 450
#extension GL_GOOGLE_include_directive : require
#include "../PBR_helpers.glsl"
#include "comp_input_structures.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

// Directional shadow with 3x3 PCF; returns visibility (1 = lit, 0 = shadowed)
float shadowVisibility(vec3 worldPos, float nDotL)
{
    if (sceneData.shadowParams.y < 0.5) // shadows disabled
        return 1.0;

    vec4 lc = sceneData.sunViewProj * vec4(worldPos, 1.0);
    vec3 proj = lc.xyz / lc.w;
    vec2 uv = proj.xy * 0.5 + 0.5;

    // outside the light frustum -> treat as lit
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0)
        return 1.0;

    // slope-scaled bias to fight shadow acne on grazing surfaces
    float bias = max(sceneData.shadowParams.x * (1.0 - nDotL), sceneData.shadowParams.x * 0.2);
    float current = proj.z;

    float vis = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
        {
            float closest = texture(shadowMap, uv + vec2(x, y) * texel).r;
            vis += (current - bias > closest) ? 0.0 : 1.0;
        }
    return vis / 9.0;
}

void main()
{
    vec3 position = texture(inPosition, vec2(inUV.s, inUV.t)).xyz;
    vec3 normal = texture(inNormal, vec2(inUV.s, inUV.t)).xyz;
    vec3 albedo = texture(inAlbedo, vec2(inUV.s, inUV.t)).xyz;
    vec2 metallicRoughness = texture(inMetalllicRoughness, vec2(inUV.s, inUV.t)).xy;

    float metallic = metallicRoughness.x;
    float roughness = metallicRoughness.y;

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0f);

    vec3 lightVec, halfwayVec, radiance, F, specular, kS, kD;
    float dist, attenuation, NDF, G, nDotL;
    float ao = 1;

    vec3 viewVec = (normalize(sceneData.cameraPos.xyz - position)).xyz;

    for (int i = 0; i < lightData.numLights; i++)
    {
        PointLight currLight = lightData.pointLights[i];
        vec3 lightPos = currLight.transform[3].xyz;
        vec3 lightDistVec = lightPos - position;
        dist = length(lightDistVec);

        if (dist > currLight.range)
            continue;

        lightVec = lightDistVec / dist;

        halfwayVec = normalize(viewVec + lightVec);

        attenuation = 1.0 / (dist * dist);
        radiance = currLight.color * attenuation * currLight.intensity;

        NDF = DistributionGGX(normal, halfwayVec, roughness);
        G = GeometrySmith(normal, viewVec, lightVec, roughness);
        F = FresnelSchlick(clamp(dot(halfwayVec, viewVec), 0.0f, 1.0f), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(normal, viewVec), 0.0f) * max(dot(normal, lightVec), 0.0f) + 0.001;
        specular = numerator / denominator;

        kS = F; // specular coefficient is equal to fresnel
        kD = vec3(1.0f) - kS;
        kD *= 1.0 - metallic;

        nDotL = max(dot(normal, lightVec), 0.0f);

        Lo += (kD * albedo / PI + specular) * radiance * nDotL;
    }

    // directional sun: sunlightDirection.xyz points sun->scene, .w = intensity
    {
        vec3 L = normalize(-sceneData.sunlightDirection.xyz);
        vec3 sunRadiance = sceneData.sunlightColor.rgb * sceneData.sunlightDirection.w;
        vec3 H = normalize(viewVec + L);

        float sNDF = DistributionGGX(normal, H, roughness);
        float sG = GeometrySmith(normal, viewVec, L, roughness);
        vec3 sF = FresnelSchlick(clamp(dot(H, viewVec), 0.0f, 1.0f), F0);

        vec3 sNum = sNDF * sG * sF;
        float sDenom = 4.0 * max(dot(normal, viewVec), 0.0f) * max(dot(normal, L), 0.0f) + 0.001;
        vec3 sSpec = sNum / sDenom;

        vec3 sKD = (vec3(1.0f) - sF) * (1.0f - metallic);
        float sNdotL = max(dot(normal, L), 0.0f);

        float shadow = shadowVisibility(position, sNdotL);
        Lo += (sKD * albedo / PI + sSpec) * sunRadiance * sNdotL * shadow;
    }

    // hemispheric ambient: sky tint from above, dimmer ground tint from below.
    vec3 skyAmbient = sceneData.ambientColor.rgb;
    vec3 groundAmbient = sceneData.ambientColor.rgb * 0.25f;
    float hemi = clamp(normal.y * 0.5f + 0.5f, 0.0f, 1.0f);
    vec3 ambient = mix(groundAmbient, skyAmbient, hemi) * albedo * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = ACESFilm(color);
    // gamma correct
    color = pow(color, vec3(1.0 / 2.2));

    outFragColor = vec4(color, 1.0);
}
