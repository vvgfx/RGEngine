#version 450
#extension GL_GOOGLE_include_directive : require
#include "../PBR_helpers.glsl" // PI, ACESFilm

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SkyParams
{
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 sunDirection; // xyz: from sun toward scene; w: sun intensity (for the disk)
    vec4 sunColor;     // rgb: sun color; w: sun disk angular size
    vec4 horizon;      // gradient colors
    vec4 zenith;
    vec4 ground;
    ivec4 mode; // x: 0 = procedural, 1 = equirect texture
}
sky;

layout(set = 1, binding = 0) uniform sampler2D normalTex; // G-buffer normal (geometry mask)
layout(set = 1, binding = 1) uniform sampler2D skyTex;    // equirectangular sky (texture mode)

vec3 rayDir(vec2 uv)
{
    vec4 clip = vec4(uv * 2.0f - 1.0f, 1.0f, 1.0f);
    vec4 world = sky.invViewProj * clip;
    world /= world.w;
    return normalize(world.xyz - sky.cameraPos.xyz);
}

void main()
{
    // Only shade background pixels: geometry wrote a unit normal, sky did not.
    vec3 n = texture(normalTex, inUV).xyz;
    if (dot(n, n) > 0.25f)
        discard;

    vec3 dir = rayDir(inUV);
    vec3 col;

    if (sky.mode.x == 1)
    {
        float u = atan(dir.z, dir.x) / (2.0f * PI) + 0.5f;
        float v = acos(clamp(dir.y, -1.0f, 1.0f)) / PI;
        col = texture(skyTex, vec2(u, v)).rgb;
    }
    else
    {
        float t = clamp(dir.y, -1.0f, 1.0f);
        col = (t >= 0.0f) ? mix(sky.horizon.rgb, sky.zenith.rgb, t)
                          : mix(sky.horizon.rgb, sky.ground.rgb, clamp(-t * 4.0f, 0.0f, 1.0f));

        vec3 toSun = normalize(-sky.sunDirection.xyz);
        float d = max(dot(dir, toSun), 0.0f);
        float size = max(sky.sunColor.w, 0.0001f);
        float disk = smoothstep(1.0f - size, 1.0f - size * 0.3f, d);
        float glow = pow(d, 64.0f) * 0.4f;
        col += sky.sunColor.rgb * (disk + glow) * max(sky.sunDirection.w, 1.0f);
    }

    // match the composite pass output (ACES tonemap + gamma)
    col = ACESFilm(col);
    col = pow(col, vec3(1.0f / 2.2f));
    outColor = vec4(col, 1.0f);
}
