# Features

Deferred renderer on a rendergraph rebuilt every frame. Linear HDR throughout; tonemap and gamma
happen once, at the end. Reverse-Z everywhere: `glm::perspective` takes near/far swapped, depth
clears to 0, compares `GREATER_OR_EQUAL`.

Scene: `assets/bistro_glb/bistro.glb` — Bistro exterior, metallic-roughness, 110 generated lights.

## Rendering

| Feature | Where |
|---|---|
| G-buffer + composite deferred shading | `DeferredRenderingFeature` |
| Cascaded shadows: 4 cascades, 4096² atlas, sphere fit, texel snapping | `ShadowFeature` |
| Point-light cube shadows, 6 faces per light, 12 casters | `LocalShadowFeature` |
| Tiled light culling: 16×16 tiles, per-tile depth bounds | `LightCullFeature` |
| Equirectangular HDRI sky, gradient fallback when absent | `lighting/sky.glsl`, `hdri_loader` |
| Cosine-convolved irradiance map for diffuse ambient | `hdri_loader` |
| One sky feeds background, ambient, transparent and SSR | `lighting/sky.glsl` |
| SSAO from position/normal G-buffer | `comp.frag` |
| Screen-space reflections, roughness-gated | `SSRFeature` |
| Transparent forward pass, same tile light list | `DeferredRenderingFeature` |
| Emissive, bloom, ACES tonemap, FXAA | `PostProcessFeature` |

## Materials — metallic-roughness only, no second path

| Feature | Where |
|---|---|
| glTF metal-rough: roughness in G, metallic in B | `mrt.frag` |
| Normal mapping via screen-space cotangent frame (Schueler) | `mrt.frag` |
| Alpha masking, per-material cutoff in `extra0.x` | `mrt.frag` |
| Emissive textures plus `KHR_materials_emissive_strength` | `MaterialSystem`, `vk_loader.cpp` |
| Trilinear plus 16× anisotropic filtering, mips on every path | `vk_engine.cpp`, `vk_loader.cpp` |
| `MSFT_texture_dds` honoured via `Texture::ddsImageIndex` | `vk_loader.cpp` |
| BC7 DDS loader with full mip chain | `dds_loader` |

## Tooling

| Feature | Where |
|---|---|
| Light debug: wireframe spheres at each light's range | `LightDebugFeature` |
| Sun gizmo driven in view space, seeded from glTF | `RGEngine.cpp`, `third_party/imGuIZMO` |
| Per-pass GPU/CPU timings, draw and triangle counts | `Rendergraph`, ImGui panel |
| Press P to dump camera position, pitch, yaw | `camera.cpp` |
| Debug views: albedo, normal, SSAO, shadow, cascade, metallic | `comp.frag` |
| Generate lights from emissive geometry, vertex-clustered | `tools/lights_from_emissive.py` |
| Transplant lights between scenes, measured XZ alignment | `tools/transplant_lights.py` |

Defaults boot into the tuned night look; daylight is one dial.

DDGI is implemented but parked — `DDGIFeature`, `AccelStructure`, `disabled_shaders/ddgi` are out
of the build.

## Performance

| Change | Effect |
|---|---|
| Tile-binned lights instead of looping all lights per pixel | Shading stops scaling with scene light count |
| Per-face cube-shadow culling, caster hits 1–2 of 6 | ~6× fewer shadow draws |
| Reject lights whose whole reach is offscreen | Most lamps skipped per frame |
| Shadow-slot hysteresis, 35% margin before eviction | Stops shadows popping while panning |
| Read light members individually, not a 96-byte copy | Transparent forward **9.0 → 1.69 ms** |
| Bloom emits only the excess over threshold | Fixed emissives blowing out neighbours |
| Half-res bloom chain | Bloom is low-frequency; full res wasted |
| Skip images no texture references | 343 dead file opens removed |
| Dropped unused 8× MSAA targets | **~354 MB** VRAM never written |

Point shadows went from **+6.0 ms GPU / +3.4 ms CPU** to enabled by default:
**10.78 → 8.73 ms GPU, 5.05 → 3.54 ms CPU** at 12 casters.

## Correctness fixes that cost frame time

| Fix | Why |
|---|---|
| Cascade corners from camera basis, not NDC unproject | Cascade 0 radius was **6198** instead of 7.56 |
| `set_shaders` no longer forces a fragment stage | Depth-only shadow pipelines need none |
| `BarrierMerger` infers depth aspect from any depth layout | Sampled shadow maps sit in `DEPTH_READ_ONLY` |
| Transparent pass emits linear HDR, never tonemaps | It was tonemapping an already-tonemapped target |
| Vertex normals use inverse-transpose model matrix | Raw matrix skewed them under non-uniform scale |
| Camera speed 500 → 10, far plane comment corrected | Bistro is metres, ~130 units, not centimetres |
| Transparent ambient was flat `vec3(0.03)`, now the sky | Glass and foliage ignored the sky entirely |
| SSR returns sky on a miss, not black | Upward-facing surfaces reflected nothing |
| Ambient reads a convolved map, never a radiance mip | A mip kept the sun at 81; irradiance peaks at 2.3 |
| Shader builds track their `#include`s via DEPFILE | Editing a `.glsl` left stale SPIR-V, silently |
| SSR separates hit confidence from surface reflectivity | One mask for both rimmed every silhouette with sky |
| Bloom bright pass is clamped | A 75,000 sky texel became a blazing fringe, not a glow |

## Gotchas

Things that cost real time and are invisible from reading the code.

### Rendergraph

- **`ReadsBuffer` / `WritesBuffer` are no-ops** (`Rendergraph.cpp:422`). Buffer handoffs between
  passes need a manual `vkCmdPipelineBarrier2` — the graph emits nothing.
- **Issue that barrier at the end of the producing pass**, not the start of the consumer. Graphics
  passes are wrapped in `vkCmdBeginRendering`, where a pipeline barrier is illegal.
- **`Build()` resets every tracked image to `UNDEFINED` each frame** (`:137`), so the first
  transition discards contents. Nothing survives between frames without changing that.
- **Naming an untracked image in a setup lambda** throws `std::out_of_range` from `Run`, not from
  the setup — the stack trace points at the wrong place.
- **Pass order is declaration order.** A pass that must sit *between* two passes of one feature has
  to be declared by that feature; feature-level ordering cannot express it. See `RegisterCullPass`.

### Build

- **Shader `#include`s need `DEPFILE`.** Without it only the `.frag`/`.vert`/`.comp` is a
  dependency, so editing a shared `.glsl` leaves stale SPIR-V behind and the build still reports
  success. This silently defeated several fixes before it was found.
- **`--target-env vulkan1.3` is not optional.** Some extensions compile into a malformed module
  without it, with no error.
- Deleting a globbed shader needs a re-configure; `CONFIGURE_DEPENDS` handles it.

### Vulkan

- **`create_image(data, ...)` takes `bytesPerTexel`, defaulting to 4.** Uploading a float format
  without passing 8 or 16 copies a fraction of the data and yields garbage, silently.
- **Reverse-Z everywhere**: near/far swapped into `glm::perspective`, depth clears to 0, compares
  `GREATER_OR_EQUAL`. Any new depth pipeline must match or it renders nothing.

### glTF and assets

- **`MSFT_texture_dds` must be enabled on the parser** or `Texture::ddsImageIndex` stays empty and
  every texture silently resolves to a PNG the asset may not ship.
- **`KHR_materials_emissive_strength` multiplies emissive by up to 100** in Bistro. Enabling it is
  correct and changes brightness enormously; nothing else is rescaled for you.
- **Bistro is metres, ~130 units across** — not centimetres. Stale comments claimed otherwise and
  poisoned the camera speed and far plane.
- **`sunlightDirection` and `sunlightColor` are repurposed**: `.xyz/.w` is sun direction and
  strength, and `sunlightColor.xy` is HDRI intensity and yaw. Both were dead fields.

### Third-party

- **imGuIZMO forces `using namespace glm;`** into the global namespace on the GLM path.
  `VGM_DISABLE_AUTO_NAMESPACE` does **not** fix it — the library's own declarations use unqualified
  `vec3`/`quat`, so defining it fails to compile. Include `imGuIZMOquat.h` last instead.
- **`IMGUI_DEFINE_MATH_OPERATORS` must be defined before `imgui.h`.** `imgui_internal.h`
  hard-errors otherwise, and imGuIZMO pulls it in. Set on the imgui target so order stops mattering.

## Sky — how and why

One shared `shaders/lighting/sky.glsl` (72 lines), called by the background, the ambient term, the
transparent pass and the SSR miss path, so sky and derived lighting cannot disagree.

**No skybox mesh and no separate pass.** It is a branch in the composite, taken where
`positionSample.w < 0.5` — the G-buffer's "geometry landed here" flag. That is a depth test using
data already fetched, so it beats a cube at the far plane: no draw call, no vertex data, and sky
pixels early-out of the whole light loop. The usual objection to screen-space sky — that you shade
every pixel — does not apply when the pass runs anyway.

**An equirectangular `.hdr`** at `assets/sky.hdr`, loaded with `stbi_loadf` and stored as RGBA16F.
Hardcoded like the scene path: core glTF 2.0 carries neither HDR nor cubemaps, and the one
extension that does (`EXT_lights_image_based`) is unsupported by fastgltf. Missing is not fatal —
the sky falls back to a tinted three-band gradient.

### The two maps, and why the split matters

| Map | Size | Used by |
|---|---|---|
| `skyHDRI` | full res, mipped | background, SSR reflections |
| `skyIrradiance` | 64×32, cosine-convolved | diffuse ambient, and only this |

**Do not collapse these.** Sampling a high mip of the radiance map as ambient looks reasonable and
is wrong by orders of magnitude: a mip is a small-angle box blur, while irradiance is a
cosine-weighted integral over the whole hemisphere. Measured on the current sky — radiance peaks at
**75,264** and mip 6 still returned **81**, against a correctly convolved peak of **2.26**. Any
surface facing the sun was blasted, which showed as bright rims wherever a silhouette swept its
normal past it.

The convolution is brute force over a 32×16 downsample: for each output direction, integrate every
input direction weighted by `cos(angle) × solidAngle`. 64k taps, instant at load. Chosen over a
spherical-harmonic projection because it is the definition of irradiance written out, with no band
coefficients to take on trust.

Still missing for real IBL: a prefiltered specular chain and the split-sum BRDF LUT. Reflections
currently sample mip 0 regardless of roughness.

### References

- [Scratchapixel — Simulating the Colors of the Sky](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html) — the physics behind sky radiance
- [Harry Alisavakis — Sky shader](https://halisavakis.com/my-take-on-shaders-sky-shader/) — complete, copyable gradient-sky shader if the HDRI is ever dropped
- [Kelvin van Hoorn — Unity skybox shader](https://kelvinvanhoorn.com/tutorials/unity_skybox_shader/) — gradient, sun disc, moon, stars, with full source
- [EXT_lights_image_based](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_lights_image_based/README.md) — how glTF *would* carry an IBL: prefiltered cubemap mips plus SH irradiance. Unsupported by fastgltf
- [State-of-the-art skybox rendering](https://gamedev.net/forums/topic/706994-state-of-the-art-skybox-rendering/) — fullscreen pass vs cube mesh
