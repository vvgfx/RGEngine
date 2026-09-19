#ifndef DDGI_COMMON_GLSL
#define DDGI_COMMON_GLSL

const float DDGI_PI = 3.14159265359;

// Interior texels per probe, plus a 1-texel border so bilinear taps at a tile edge wrap across the
// octahedral seam instead of bleeding into the neighbouring probe.
const int DDGI_IRR_INTERIOR = 6;
const int DDGI_DIST_INTERIOR = 14;
const int DDGI_IRR_TILE = DDGI_IRR_INTERIOR + 2;
const int DDGI_DIST_TILE = DDGI_DIST_INTERIOR + 2;

// Packed as vec4 rows so std140 layout needs no explicit padding.
layout(set = DDGI_SET, binding = 0) uniform DDGIVolumeBlock
{
    vec4 origin;   // xyz = world position of probe (0,0,0)
    vec4 spacing;  // xyz = world units between probes
    ivec4 counts;  // xyz = probe counts, w = rays per probe
    vec4 blend;    // x hysteresis, y normalBias, z viewBias, w maxRayDistance
    vec4 misc;     // x depthSharpness, y irradianceGamma, z frameIndex, w unused
    vec4 skyColor; // zenith colour used when a probe ray escapes the scene
    vec4 flags;    // x = ray-traced shadows in the composite pass
}
ddgi;

#define DDGI_ORIGIN ddgi.origin.xyz
#define DDGI_SPACING ddgi.spacing.xyz
#define DDGI_COUNTS ddgi.counts.xyz
#define DDGI_RAYS ddgi.counts.w
#define DDGI_HYSTERESIS ddgi.blend.x
#define DDGI_NORMAL_BIAS ddgi.blend.y
#define DDGI_VIEW_BIAS ddgi.blend.z
#define DDGI_MAX_DISTANCE ddgi.blend.w
#define DDGI_DEPTH_SHARPNESS ddgi.misc.x
#define DDGI_GAMMA ddgi.misc.y

int ddgiProbeCount()
{
    return DDGI_COUNTS.x * DDGI_COUNTS.y * DDGI_COUNTS.z;
}

int ddgiProbeIndex(ivec3 c)
{
    return c.x + c.y * DDGI_COUNTS.x + c.z * DDGI_COUNTS.x * DDGI_COUNTS.y;
}

ivec3 ddgiProbeCoord(int idx)
{
    int nx = DDGI_COUNTS.x;
    int ny = DDGI_COUNTS.y;
    return ivec3(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}

vec3 ddgiProbePosition(ivec3 c)
{
    return DDGI_ORIGIN + vec3(c) * DDGI_SPACING;
}

// Probes are packed into a flat 2D atlas: (Nx*Ny) tiles across, Nz tiles down.
ivec2 ddgiTileOrigin(int probeIdx, int tileSize)
{
    int tilesPerRow = DDGI_COUNTS.x * DDGI_COUNTS.y;
    return ivec2(probeIdx % tilesPerRow, probeIdx / tilesPerRow) * tileSize;
}

// Cigolle et al. octahedral mapping: sphere -> unit square, low distortion and trivial borders.
vec2 octEncode(vec3 v)
{
    float l1 = abs(v.x) + abs(v.y) + abs(v.z);
    vec3 n = v / max(l1, 1e-8);
    vec2 r = n.xy;
    if (n.z < 0.0)
    {
        r = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return r; // [-1, 1]
}

vec3 octDecode(vec2 f)
{
    vec3 v = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (v.z < 0.0)
    {
        v.xy = (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(v);
}

/// Direction encoded by an interior texel, where local is in [0, interior).
vec3 ddgiTexelDirection(ivec2 local, int interior)
{
    vec2 oct = (vec2(local) + 0.5) / float(interior) * 2.0 - 1.0;
    return octDecode(oct);
}

/// Atlas UV for a direction, landing inside the tile interior so the border is only ever a filter tap.
vec2 ddgiProbeUV(int probeIdx, vec3 dir, int tileSize, vec2 atlasSize)
{
    vec2 oct = octEncode(normalize(dir)) * 0.5 + 0.5;
    vec2 texel = vec2(ddgiTileOrigin(probeIdx, tileSize)) + 1.0 + oct * float(tileSize - 2);
    return texel / atlasSize;
}

/// Evenly distributed directions; rotating the whole set per frame is what makes hysteresis converge.
vec3 sphericalFibonacci(float i, float n)
{
    const float PHI = 1.6180339887498948482;
    float phi = 2.0 * DDGI_PI * fract(i * (PHI - 1.0));
    float cosTheta = 1.0 - (2.0 * i + 1.0) / n;
    float sinTheta = sqrt(clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0));
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

/**
 * A cheap three-band sky. Used both as the visible background and as the miss radiance for probe
 * rays, so the lighting and the backdrop can never disagree.
 */
vec3 ddgiSky(vec3 dir)
{
    vec3 zenith = ddgi.skyColor.rgb;
    vec3 horizon = ddgi.skyColor.rgb * 2.2 + vec3(0.05);
    vec3 ground = ddgi.skyColor.rgb * 0.25;

    float d = clamp(dir.y, -1.0, 1.0);

    if (d < 0.0)
    {
        // pull the horizon band tight so the ground does not wash out the lower hemisphere
        return mix(horizon, ground, clamp(-d * 4.0, 0.0, 1.0));
    }
    return mix(horizon, zenith, clamp(pow(d, 0.45), 0.0, 1.0));
}

#endif
