#version 450
#extension GL_GOOGLE_include_directive : require
#include "../PBR_helpers.glsl"
#include "comp_input_structures.glsl"

// the DDGI descriptor set layout is shared with the compute passes, where it binds at set 0
#define DDGI_SET 3
#include "../ddgi/ddgi_common.glsl"
#include "../ddgi/ddgi_sample.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

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

    // DDGISampleIrradiance returns average incident radiance, so albedo multiplies it directly.
    // misc.w is the runtime toggle; falling back to the old constant keeps the scene lit when off.
    vec3 ambient = (ddgi.misc.w > 0.5) ? albedo * DDGISampleIrradiance(position, normal, viewVec) * ao : vec3(0.03f) * albedo * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = ACESFilm(color);
    // gamma correct
    color = pow(color, vec3(1.0 / 2.2));

    outFragColor = vec4(color, 1.0);
}
