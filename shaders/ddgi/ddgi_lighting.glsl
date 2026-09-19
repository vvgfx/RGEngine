#ifndef DDGI_LIGHTING_GLSL
#define DDGI_LIGHTING_GLSL

// Mirrors DeferredRenderingFeature::LightData.
struct DDGIPointLight
{
    mat4 transform;
    vec3 color;
    float intensity;
    float range;
};

layout(set = DDGI_SET, binding = 6) uniform DDGILightBlock
{
    DDGIPointLight pointLights[25];
    int numLights;
}
ddgiLights;

vec3 ddgiSky(vec3 dir)
{
    // Skipping this makes exteriors far too dark; the sky is the dominant indirect source outdoors.
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(ddgi.skyColor.rgb * 0.3, ddgi.skyColor.rgb, t);
}

/// Direct irradiance at a probe-ray hit point. shadowed=true spends one occlusion ray per contributing light.
vec3 ddgiDirectIrradiance(vec3 P, vec3 N, bool shadowed)
{
    vec3 E = vec3(0.0);

    for (int i = 0; i < ddgiLights.numLights; i++)
    {
        DDGIPointLight L = ddgiLights.pointLights[i];
        vec3 toLight = L.transform[3].xyz - P;
        float dist = length(toLight);

        if (dist > L.range || dist < 1e-4)
        {
            continue;
        }

        vec3 l = toLight / dist;
        float ndl = max(dot(N, l), 0.0);
        if (ndl <= 0.0)
        {
            continue;
        }

        if (shadowed && TraceShadow(P + N * 0.02, l, dist - 0.02))
        {
            continue;
        }

        E += L.color * L.intensity * ndl / (dist * dist);
    }

    return E;
}

#endif
