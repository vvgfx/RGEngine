#version 450

#extension GL_GOOGLE_include_directive : require
#include "light_input_structures.glsl"
#include "../../PBR_helpers.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inPos;

layout(location = 0) out vec4 outFragColor;

void main()
{
    vec3 viewVec, lightVec, halfwayVec;
    vec3 tempNormal;
    float dist, attenuation;
    vec3 radiance, specular;
    vec3 kS, kD;
    float nDotL;
    float NDF;
    float G;
    vec3 F;

    tempNormal = normalize(inNormal); // world space
    // if (!gl_FrontFacing)
    //     tempNormal = -tempNormal;
    viewVec = (normalize(sceneData.cameraPos - inPos)).xyz; // world space.
    vec3 albedo = pow(inColor * texture(colorTex, vec2(inUV.s, inUV.t)).rgb, vec3(2.2)); // conversion from sRGB
    // to linear space
    vec3 normal = tempNormal;
    vec2 metalRough = texture(metalRoughTex, inUV).bg;
    float metallic = metalRough.x * materialData.metal_rough_factors.x;
    float roughness = metalRough.y * materialData.metal_rough_factors.y;
    float ao = 1;

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0f);

    for (int i = 0; i < lightData.numLights; i++)
    {
        // see comp.frag: read members individually so culled lights never pull the mat4
        if (lightData.pointLights[i].type == 0)
        {
            // directional: travels along the node's -Z, no position or falloff
            lightVec = normalize(lightData.pointLights[i].transform[2].xyz);
            radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity;
        }
        else
        {
            vec3 lightDistVec = lightData.pointLights[i].transform[3].xyz - inPos.xyz;

            float distSq = dot(lightDistVec, lightDistVec);
            float range = lightData.pointLights[i].range;
            if (distSq > range * range)
                continue;

            dist = sqrt(distSq);
            lightVec = lightDistVec / dist;
            attenuation = 1.0 / max(distSq, 1e-4);
            radiance = lightData.pointLights[i].color * attenuation * lightData.pointLights[i].intensity;
        }

        halfwayVec = normalize(viewVec + lightVec);

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

    vec3 ambient = vec3(0.03f) * albedo * ao;

    // linear HDR: the post pass owns tonemapping and gamma
    outFragColor = vec4(ambient + Lo, 1.0);
}
