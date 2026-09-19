#ifndef DDGI_LIGHTING_GLSL
#define DDGI_LIGHTING_GLSL

// Mirrors DeferredRenderingFeature::LightData.
struct DDGIPointLight
{
    mat4 transform;
    vec3 color;
    float intensity;
    float range;
    int type; // 0 = directional, 1 = spot, 2 = point
};

layout(set = DDGI_SET, binding = 6) uniform DDGILightBlock
{
    DDGIPointLight pointLights[128];
    int numLights;
}
ddgiLights;

/// Direct irradiance at a probe-ray hit point. shadowed=true spends one occlusion ray per contributing light.
vec3 ddgiDirectIrradiance(vec3 P, vec3 N, bool shadowed)
{
    vec3 E = vec3(0.0);

    for (int i = 0; i < ddgiLights.numLights; i++)
    {
        DDGIPointLight L = ddgiLights.pointLights[i];

        vec3 l;
        float dist;
        vec3 radiance;

        if (L.type == 0)
        {
            // directional: travels along the node's -Z, so the vector towards it is +Z
            l = normalize(L.transform[2].xyz);
            dist = DDGI_MAX_DISTANCE;
            radiance = L.color * L.intensity;
        }
        else
        {
            vec3 toLight = L.transform[3].xyz - P;
            dist = length(toLight);

            if (dist > L.range || dist < 1e-4)
            {
                continue;
            }

            l = toLight / dist;
            radiance = L.color * L.intensity / (dist * dist);
        }

        float ndl = max(dot(N, l), 0.0);
        if (ndl <= 0.0)
        {
            continue;
        }

        if (shadowed && TraceShadow(P + N * 0.02, l, dist - 0.02))
        {
            continue;
        }

        E += radiance * ndl;
    }

    return E;
}

#endif
