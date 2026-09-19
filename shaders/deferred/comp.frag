#version 450

#extension GL_GOOGLE_include_directive : require
#include "../PBR_helpers.glsl"
#include "comp_input_structures.glsl"

#define SHADOW_SET 3
#include "../shadow/shadow_input.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

/// Sky used both as the visible background and as the ambient source, so they cannot disagree.
vec3 skyColor(vec3 dir)
{
    vec3 zenith = sceneData.ambientColor.rgb;
    vec3 horizon = sceneData.ambientColor.rgb * 2.2 + vec3(0.05);
    vec3 ground = sceneData.ambientColor.rgb * 0.25;

    float d = clamp(dir.y, -1.0, 1.0);
    if (d < 0.0)
    {
        return mix(horizon, ground, clamp(-d * 4.0, 0.0, 1.0));
    }
    return mix(horizon, zenith, clamp(pow(d, 0.45), 0.0, 1.0));
}

/// World-space ray through this pixel, rebuilt from the inverse view-projection.
vec3 viewRay(vec2 uv)
{
    vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = sceneData.invViewproj * ndc;
    return normalize(world.xyz / world.w - sceneData.cameraPos.xyz);
}

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/**
 * Screen-space ambient occlusion from the position/normal G-buffer.
 *
 * Without this the sky term floods every crevice equally and the image reads flat; this is what
 * puts contact darkening under awnings, in doorways and at wall bases.
 */
float computeSSAO(vec3 P, vec3 N)
{
    int sampleCount = int(sceneData.ssaoParams.x);
    if (sampleCount <= 0)
    {
        return 1.0;
    }

    float radius = sceneData.ssaoParams.y;
    float strength = sceneData.ssaoParams.z;

    // per-pixel rotation decorrelates the sample pattern so banding becomes noise
    float angle = hash12(gl_FragCoord.xy) * 6.2831853;
    float ca = cos(angle);
    float sa = sin(angle);

    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    float occlusion = 0.0;

    for (int i = 0; i < sampleCount; i++)
    {
        // golden-angle spiral in the hemisphere, biased towards the centre
        float fi = (float(i) + 0.5) / float(sampleCount);
        float r = sqrt(fi);
        float phi = fi * 20.0 + angle;

        vec2 disk = vec2(cos(phi), sin(phi)) * r;
        disk = vec2(disk.x * ca - disk.y * sa, disk.x * sa + disk.y * ca);

        vec3 dir = normalize(T * disk.x + B * disk.y + N * 0.6);
        vec3 samplePos = P + dir * radius * (0.3 + 0.7 * fi);

        vec4 clip = sceneData.viewproj * vec4(samplePos, 1.0);
        if (clip.w <= 0.0)
        {
            continue;
        }
        vec2 sampleUV = (clip.xy / clip.w) * 0.5 + 0.5;
        if (any(lessThan(sampleUV, vec2(0.0))) || any(greaterThan(sampleUV, vec2(1.0))))
        {
            continue;
        }

        vec4 gb = texture(inPosition, sampleUV);
        if (gb.w < 0.5)
        {
            continue; // sky
        }

        // Is the surface actually in front of our sample point, and close enough to matter?
        vec3 toSurface = gb.xyz - P;
        float surfaceDist = length(gb.xyz - sceneData.cameraPos.xyz);
        float sampleDist = length(samplePos - sceneData.cameraPos.xyz);

        if (surfaceDist < sampleDist - 0.02)
        {
            // range check stops distant geometry from darkening foreground silhouettes
            float rangeFade = clamp(radius / max(length(toSurface), 1e-4), 0.0, 1.0);
            occlusion += clamp(dot(normalize(toSurface), N), 0.0, 1.0) * rangeFade;
        }
    }

    return clamp(1.0 - (occlusion / float(sampleCount)) * strength, 0.0, 1.0);
}

/// Shadow factor for the first directional light, isolated for the debug view.
float shadowOnly(vec3 P, float viewDepth, vec3 N)
{
    for (int i = 0; i < lightData.numLights; i++)
    {
        if (lightData.pointLights[i].type == 0)
        {
            return sampleShadow(P, viewDepth, N, normalize(lightData.pointLights[i].transform[2].xyz));
        }
    }
    return 1.0;
}

/// Which cascade a pixel lands in: red, green, blue, yellow from near to far.
vec3 cascadeTint(float viewDepth)
{
    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++)
    {
        if (viewDepth < shadowData.cascadeSplits[i])
        {
            return i == 0 ? vec3(1, 0, 0) : i == 1 ? vec3(0, 1, 0) : i == 2 ? vec3(0, 0, 1) : vec3(1, 1, 0);
        }
    }
    return vec3(0.2);
}

void main()
{
    vec4 positionSample = texture(inPosition, inUV);

    // The G-buffer clears position to 0 and mrt.frag writes w = 1, so w marks "geometry here".
    if (positionSample.w < 0.5)
    {
        outFragColor = vec4(skyColor(viewRay(inUV)), 1.0);
        return;
    }

    vec3 position = positionSample.xyz;
    vec3 normal = normalize(texture(inNormal, inUV).xyz);
    vec3 albedo = texture(inAlbedo, inUV).xyz;
    vec2 metallicRoughness = texture(inMetalllicRoughness, inUV).xy;

    float metallic = metallicRoughness.x;
    float roughness = metallicRoughness.y;

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 Lo = vec3(0.0f);

    vec3 viewVec = normalize(sceneData.cameraPos.xyz - position);
    float viewDepth = length(position - sceneData.cameraPos.xyz);

    for (int i = 0; i < lightData.numLights; i++)
    {
        // Read members individually: copying the struct pulls its mat4 for every light, and almost
        // all of them are about to fail the range test anyway.
        vec3 lightVec;
        vec3 radiance;

        if (lightData.pointLights[i].type == 0)
        {
            // glTF directional lights shine along the node's -Z, so the vector towards it is +Z.
            lightVec = normalize(lightData.pointLights[i].transform[2].xyz);
            radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity;
            radiance *= sampleShadow(position, viewDepth, normal, lightVec);
        }
        else
        {
            vec3 lightDistVec = lightData.pointLights[i].transform[3].xyz - position;

            // squared compare: rejects without a sqrt, and touches only one matrix column
            float distSq = dot(lightDistVec, lightDistVec);
            float range = lightData.pointLights[i].range;
            if (distSq > range * range)
                continue;

            lightVec = lightDistVec / sqrt(distSq);
            radiance = lightData.pointLights[i].color * lightData.pointLights[i].intensity / max(distSq, 1e-4);
        }

        float nDotL = max(dot(normal, lightVec), 0.0f);
        if (nDotL <= 0.0)
            continue;

        vec3 halfwayVec = normalize(viewVec + lightVec);

        float NDF = DistributionGGX(normal, halfwayVec, roughness);
        float G = GeometrySmith(normal, viewVec, lightVec, roughness);
        vec3 F = FresnelSchlick(clamp(dot(halfwayVec, viewVec), 0.0f, 1.0f), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(normal, viewVec), 0.0f) * nDotL + 0.001;
        vec3 specular = numerator / denominator;

        vec3 kD = (vec3(1.0f) - F) * (1.0 - metallic);

        Lo += (kD * albedo / PI + specular) * radiance * nDotL;
    }

    float ao = computeSSAO(position, normal);

    int debugMode = int(sceneData.debugParams.x);
    if (debugMode > 0)
    {
        vec3 dbg;
        if (debugMode == 1)
            dbg = albedo;
        else if (debugMode == 2)
            dbg = normal * 0.5 + 0.5;
        else if (debugMode == 3)
            dbg = vec3(ao);
        else if (debugMode == 4)
            dbg = vec3(shadowOnly(position, viewDepth, normal));
        else if (debugMode == 5)
            dbg = cascadeTint(viewDepth);
        else if (debugMode == 6)
            dbg = vec3(roughness);
        else if (debugMode == 7)
            dbg = vec3(metallic);
        else if (debugMode == 8)
            dbg = fract(abs(position) * 0.1);
        else
            // raw shadow atlas, all four cascades. Black = cleared (no caster), grey/white = depth.
            dbg = vec3(texture(shadowAtlasRaw, inUV).r);

        // the post pass still tonemaps this, which is fine: the mapping stays monotonic and these
        // views only need to be readable, not colour-accurate
        outFragColor = vec4(clamp(dbg, 0.0, 1.0), 1.0);
        return;
    }

    // Hemispheric sky ambient: the upper hemisphere sees sky, the lower sees bounced ground. Coarse
    // compared to a probe field, but it is occluded by AO and costs nothing.
    vec3 ambient = albedo * mix(skyColor(-normal) * 0.35, skyColor(normal), 0.5) * ao * sceneData.ssaoParams.w;

    outFragColor = vec4(ambient + Lo, 1.0);
}
