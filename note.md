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
| Preetham analytic sky, blended to a tinted night gradient | `lighting/sky.glsl` |
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
| Sky day/night blend keyed on sun strength, not just elevation | A high sun at 0.02 intensity is still night |

## Sky — how and why

One shared `shaders/lighting/sky.glsl`, called by the background, the ambient term, the transparent
pass and the SSR miss path, so sky and derived lighting cannot disagree.

**No skybox mesh and no separate pass.** It is a branch in the composite, taken where
`positionSample.w < 0.5` — the G-buffer's "geometry landed here" flag. That is a depth test using
data already fetched, so it beats a cube at the far plane: no draw call, no vertex data, and sky
pixels early-out of the whole light loop. The usual objection to screen-space sky, that you shade
every pixel, does not apply when the pass runs anyway.

**Daylight is Preetham**: a Perez distribution fitted to turbidity, evaluated in xyY, converted to
linear sRGB, plus a 0.53° sun disc. Preetham rather than Hosek-Wilkie because HW needs a ~1500-float
fitted coefficient dataset where Preetham derives its coefficients from turbidity with linear fits.

**Below the horizon it blends to the old three-band tinted gradient.** Preetham is a daylight model
— its zenith luminance goes negative once the sun crosses the horizon — and the gradient is what
the lamps and point shadows were balanced against.

Known weak spot: Preetham's sunsets, and low sun angles generally — see the Zotti & Wilkie review
below. Hosek-Wilkie is the drop-in upgrade if that starts to matter; Hillaire is the step beyond,
and needs precomputed LUTs and extra passes.

**Coefficient provenance.** The 15 Perez distribution coefficients and the zenith luminance formula
in `sky.glsl` were checked line by line against Appendix A.2 of the paper and match exactly. The
`chi` expression and the two zenith *chromaticity* polynomial matrices (`xz`, `yz`) are typeset as
matrix equations that do not survive text extraction, so those remain unverified against the
primary source — worth re-checking against a printed copy if the sky's hue ever looks off.

### References

- [Preetham, Shirley & Smits 1999, *A Practical Analytic Model for Daylight*](https://courses.cs.duke.edu/cps124/spring08/assign/07_papers/p91-preetham.pdf) — coefficients are in Appendix A.2
- [Zotti & Wilkie 2007, *A Critical Review of the Preetham Skylight Model*](https://www.cg.tuwien.ac.at/research/publications/2007/zotti-2007-wscg/zotti-2007-wscg-paper.pdf) — where and why it breaks down
- [Hosek & Wilkie 2012, *An Analytic Model for Full Spectral Sky-Dome Radiance*](https://cgg.mff.cuni.cz/projects/SkylightModelling/)
- [Hillaire 2020, *A Scalable and Production Ready Sky and Atmosphere Rendering Technique*](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14050)
- [State-of-the-art skybox rendering discussion](https://gamedev.net/forums/topic/706994-state-of-the-art-skybox-rendering/)
