#ifndef DDGI_SAMPLE_GLSL
#define DDGI_SAMPLE_GLSL

// Requires ddgi_common.glsl plus these two samplers (both read the atlases in GENERAL layout).
layout(set = DDGI_SET, binding = 4) uniform sampler2D ddgiIrradianceTex;
layout(set = DDGI_SET, binding = 5) uniform sampler2D ddgiDistanceTex;

/**
 * Interpolates irradiance from the 8-probe cage around P.
 *
 * Returns average incident radiance (E/pi), so callers multiply by albedo directly.
 */
vec3 DDGISampleIrradiance(vec3 P, vec3 N, vec3 V)
{
    vec2 irrSize = vec2(textureSize(ddgiIrradianceTex, 0));
    vec2 distSize = vec2(textureSize(ddgiDistanceTex, 0));

    // Push the query point off the surface along both the normal and the view ray. This single bias
    // removes most leaking on its own by moving away from the shadow/lit discontinuity.
    vec3 biased = P + N * DDGI_NORMAL_BIAS + V * DDGI_VIEW_BIAS;

    vec3 gridF = (biased - DDGI_ORIGIN) / DDGI_SPACING;
    ivec3 baseCoord = clamp(ivec3(floor(gridF)), ivec3(0), DDGI_COUNTS - 1);
    vec3 alpha = clamp(gridF - vec3(baseCoord), vec3(0.0), vec3(1.0));

    vec3 sum = vec3(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < 8; i++)
    {
        ivec3 offset = ivec3(i, i >> 1, i >> 2) & ivec3(1);
        ivec3 coord = clamp(baseCoord + offset, ivec3(0), DDGI_COUNTS - 1);
        int probeIdx = ddgiProbeIndex(coord);

        vec3 probePos = ddgiProbePosition(coord);
        vec3 toProbe = probePos - P;
        float distToProbe = length(toProbe);
        vec3 dirToProbe = distToProbe > 1e-6 ? toProbe / distToProbe : N;

        // Trilinear term.
        vec3 tri = mix(1.0 - alpha, alpha, vec3(offset));
        float weight = tri.x * tri.y * tri.z;

        // Smooth backface term: fades probes behind the surface instead of hard-culling them, which
        // would produce visible seams at cage boundaries.
        weight *= pow(max(dot(dirToProbe, N), 0.0) * 0.5 + 0.5, 2.0) + 0.2;

        // Chebyshev visibility. The stored depth moments act as a variance shadow map, which is what
        // stops a probe inside a wall from lighting the room on the other side.
        vec3 biasedDir = biased - probePos;
        float biasedDist = length(biasedDir);
        vec2 moments = texture(ddgiDistanceTex, ddgiProbeUV(probeIdx, normalize(biasedDir), DDGI_DIST_TILE, distSize)).rg;

        float mean = moments.x;
        if (biasedDist > mean)
        {
            float variance = abs(mean * mean - moments.y);
            float d = biasedDist - mean;
            float chebyshev = variance / (variance + d * d);
            // Cubing sharpens the falloff; the max() keeps a floor so probes never vanish entirely.
            weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }

        // Crush very small weights: they are almost always faint leaks, and the eye is highly
        // sensitive to a little stray light in an otherwise dark region.
        const float crush = 0.2;
        if (weight < crush)
        {
            weight *= (weight * weight) / (crush * crush);
        }

        weight = max(weight, 1e-6);

        vec3 irr = texture(ddgiIrradianceTex, ddgiProbeUV(probeIdx, N, DDGI_IRR_TILE, irrSize)).rgb;
        irr = pow(irr, vec3(DDGI_GAMMA * 0.5)); // undo the perceptual encode (see the blend pass)

        sum += irr * weight;
        totalWeight += weight;
    }

    if (totalWeight <= 0.0)
    {
        return vec3(0.0);
    }

    return sum / totalWeight;
}

#endif
