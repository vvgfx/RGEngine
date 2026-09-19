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
| Procedural sky gradient, also sampled as ambient source | `comp.frag` |
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
