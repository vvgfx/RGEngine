#version 460

// 460 is required for the rayQueryEXT keyword; glslang gates it behind that version.
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#include "../PBR_helpers.glsl"
#include "comp_input_structures.glsl"

// the DDGI descriptor set layout is shared with the compute passes, where it binds at set 0
#define DDGI_SET 3
#include "../ddgi/ddgi_common.glsl"
#include "../ddgi/ddgi_sample.glsl"

layout(set = DDGI_SET, binding = 7) uniform accelerationStructureEXT topLevelAS;

/// One any-hit ray towards the light. Reuses the DDGI TLAS, so shadows cost no extra structure.
float sunVisibility(vec3 P, vec3 N, vec3 L, float maxDist)
{
    if (ddgi.flags.x < 0.5)
    {
        return 1.0;
    }

    // offset along the normal so a surface cannot shadow itself at grazing angles
    vec3 origin = P + N * 2.0;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.1, L, maxDist);
    while (rayQueryProceedEXT(rq))
    {
    }

    return rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

/// World-space ray through this pixel, rebuilt from the inverse view-projection.
vec3 viewRay(vec2 uv)
{
    vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = sceneData.invViewproj * ndc;
    return normalize(world.xyz / world.w - sceneData.cameraPos.xyz);
}

void main()
{
    // The G-buffer clears position to 0 and mrt.frag writes w = 1, so w marks "geometry here".
    // Those pixels are sky: the composite pass clears drawImage, so the background compute pass
    // cannot supply it.
    if (texture(inPosition, inUV).w < 0.5)
    {
        vec3 sky = ddgiSky(viewRay(inUV));
        outFragColor = vec4(pow(ACESFilm(sky), vec3(1.0 / 2.2)), 1.0);
        return;
    }

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
        // Read members individually. Copying the whole struct pulls its mat4 for every light, and
        // almost all of them are about to fail the range test anyway.
        float shadowDist;

        if (lightData.pointLights[i].type == 0)
        {
            // glTF directional lights shine along the node's -Z, so the vector towards the light is
            // +Z. They have no position, no range and no falloff.
            lightVec = normalize(lightData.pointLights[i].transform[2].xyz);
            radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity;
            shadowDist = 1e7;
        }
        else
        {
            vec3 lightDistVec = lightData.pointLights[i].transform[3].xyz - position;

            // squared compare: rejects without a sqrt, and touches only one matrix column
            float distSq = dot(lightDistVec, lightDistVec);
            float range = lightData.pointLights[i].range;
            if (distSq > range * range)
                continue;

            dist = sqrt(distSq);
            lightVec = lightDistVec / dist;
            attenuation = 1.0 / max(distSq, 1e-4);
            radiance = lightData.pointLights[i].color * attenuation * lightData.pointLights[i].intensity;
            shadowDist = dist;
        }

        radiance *= sunVisibility(position, normal, lightVec, shadowDist);
        if (radiance == vec3(0.0))
            continue;

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
