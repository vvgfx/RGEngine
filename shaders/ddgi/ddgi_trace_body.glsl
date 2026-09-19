#ifndef DDGI_TRACE_BODY_GLSL
#define DDGI_TRACE_BODY_GLSL

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = DDGI_SET, binding = 1, rgba16f) uniform writeonly image2D probeRayData;

layout(push_constant) uniform TracePushConstants
{
    mat4 rayRotation; // re-randomised every frame so hysteresis converges over many frames
    vec4 flags;       // x = trace shadow rays
}
pc;

void main()
{
    int rayIndex = int(gl_GlobalInvocationID.x);
    int probeIndex = int(gl_GlobalInvocationID.y);

    if (rayIndex >= DDGI_RAYS || probeIndex >= ddgiProbeCount())
    {
        return;
    }

    vec3 dir = normalize(mat3(pc.rayRotation) * sphericalFibonacci(float(rayIndex), float(DDGI_RAYS)));
    vec3 origin = ddgiProbePosition(ddgiProbeCoord(probeIndex));

    DDGIHit hit;
    TraceScene(origin, dir, DDGI_MAX_DISTANCE, hit);

    vec3 radiance;
    float dist;

    if (!hit.hit)
    {
        radiance = ddgiSky(dir);
        dist = DDGI_MAX_DISTANCE;
    }
    else if (hit.backface)
    {
        // Negative distance marks a backface hit. The blend pass skips these, so a probe buried in
        // geometry degrades gracefully instead of going black.
        radiance = vec3(0.0);
        dist = -abs(hit.t) * 0.2;
    }
    else
    {
        vec3 E = ddgiDirectIrradiance(hit.position, hit.normal, pc.flags.x > 0.5);

        // Sampling the probes here is what gives multi-bounce GI for free: each frame's update
        // feeds on the previous frame's converged irradiance.
        vec3 indirect = DDGISampleIrradiance(hit.position, hit.normal, -dir);

        radiance = hit.albedo * (E / DDGI_PI + indirect);
        dist = hit.t;
    }

    imageStore(probeRayData, ivec2(rayIndex, probeIndex), vec4(radiance, dist));
}

#endif
