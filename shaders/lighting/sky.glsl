#ifndef SKY_GLSL
#define SKY_GLSL

/**
 * Equirectangular HDRI sky, shared by the background, the ambient term, the transparent pass and
 * the SSR miss path -- so what you see and what lights the scene cannot disagree.
 *
 * Two maps, and the split is the whole point:
 *   skyHDRI        full-resolution radiance. Background and reflections.
 *   skyIrradiance  tiny cosine-convolved map. Diffuse ambient, and ONLY this.
 *
 * Sampling a mip of the radiance map as ambient is the trap, and it cost a long debugging session.
 * A mip is a small-angle box blur; irradiance is a cosine-weighted integral over the whole
 * hemisphere. An HDRI sun reaching 75,000 survives into mip 6 nearly intact, so any surface facing
 * it was blasted -- visible as bright rims wherever a silhouette swept its normal past the sun.
 *
 * A consumer declares both samplers at its own set/binding and defines SKY_HDRI_BOUND. With no map
 * loaded, everything falls back to a tinted gradient so the engine still runs.
 *
 * skyParams: x = intensity (0 selects the gradient), y = yaw in radians.
 */

const float SKY_PI = 3.14159265359;

/// Standard latitude/longitude layout: u wraps the horizon from -Z, v runs pole to pole.
vec2 skyEquirectUV(vec3 dir, float yaw)
{
    float c = cos(yaw);
    float s = sin(yaw);
    vec3 d = vec3(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);
    return vec2(atan(d.z, d.x) / (2.0 * SKY_PI) + 0.5, asin(clamp(d.y, -1.0, 1.0)) / SKY_PI + 0.5);
}

/// Fallback when no HDRI is loaded. Three bands from one tint; pow(0.45) keeps the bright horizon
/// band tight rather than smearing it up the dome.
vec3 skyGradient(vec3 dir, vec3 tint)
{
    vec3 horizon = tint * 2.2 + vec3(0.05);
    float d = clamp(dir.y, -1.0, 1.0);
    if (d < 0.0)
    {
        return mix(horizon, tint * 0.25, clamp(-d * 4.0, 0.0, 1.0));
    }
    return mix(horizon, tint, clamp(pow(d, 0.45), 0.0, 1.0));
}

#ifdef SKY_HDRI_BOUND
/// Radiance in a direction: the background, and what a reflection ray returns on a miss.
vec3 skyEnv(vec3 dir, vec2 skyParams, vec3 tint)
{
    if (skyParams.x <= 0.0)
    {
        return skyGradient(dir, tint);
    }
    return textureLod(skyHDRI, skyEquirectUV(dir, skyParams.y), 0.0).rgb * skyParams.x;
}

/// Diffuse ambient. Comes from the convolved map, never from a mip of the radiance map.
vec3 skyEnvAmbient(vec3 n, vec2 skyParams, vec3 tint)
{
    if (skyParams.x <= 0.0)
    {
        return skyGradient(n, tint) * 0.5;
    }
    return textureLod(skyIrradiance, skyEquirectUV(n, skyParams.y), 0.0).rgb * skyParams.x;
}
#else
#define skyEnv(dir, p, tint) skyGradient(dir, tint)
#define skyEnvAmbient(n, p, tint) (skyGradient(n, tint) * 0.5)
#endif

#endif
