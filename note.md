# Features

Deferred renderer on a rendergraph that rebuilds every frame. Linear HDR throughout; tonemap and
gamma happen once, at the end.

| Feature | Where |
|---|---|
| G-buffer + composite deferred shading | `DeferredRenderingFeature` |
| Cascaded shadow maps — 4 cascades, 2×2 in a 4096² atlas, bounding-sphere fit, texel snapping | `ShadowFeature` |
| Point-light cube shadows — 6 faces per light into a shared atlas, 12 casters by budget | `LocalShadowFeature` |
| Tiled light culling — 16×16 tiles, per-tile depth bounds, sphere test | `LightCullFeature` |
| Procedural sky — analytic zenith/horizon/ground gradient from a single tint, no cubemap or HDRI. Also sampled along the normal as the ambient source, so background and ambient cannot desync | `comp.frag` |
| SSAO | `shaders/deferred/comp.frag` |
| Screen-space reflections | `SSRFeature` |
| Transparent forward pass — blended geometry shaded after the composite, on the same tile light list | `DeferredRenderingFeature` |
| Emissive + bloom, tonemap, FXAA | `PostProcessFeature` |
| Debug views — albedo, normal, SSAO, shadow, cascade, roughness, metallic, emissive, atlas | `comp.frag` |

## Materials

| Feature | Where |
|---|---|
| BC7 DDS loading with the full mip chain | `dds_loader` |
| Trilinear + 16× anisotropic filtering | `vk_engine.cpp`, `vk_loader.cpp` |
| Normal mapping via a screen-space cotangent frame (Schueler) — no vertex tangents, so `Vertex` stays 48 B and the BLAS stride is untouched | `mrt.frag` |
| Alpha masking — per-material cutoff packed into the spare `MaterialConstants::extra0` | `mrt.frag` |
| Spec-gloss materials mapped to metallic-roughness, per-texel gloss from the alpha channel | `vk_loader.cpp`, `mrt.frag` |
| Emissive textures and factors | `MaterialSystem` |

## Tooling

| Feature | Where |
|---|---|
| Sun direction gizmo — drag a 3D arrow to aim the sun. Driven in **view space**, so the sphere stands for the screen: drag up and the sun rises in shot. Seeded from the glTF on frame 1 | `RGEngine.cpp`, `third_party/imGuIZMO` |
| Per-pass GPU/CPU timings, draw and triangle counts | `Rendergraph`, ImGui panel |
| Press **P** to dump camera position, pitch and yaw as paste-ready source | `camera.cpp` |

The sun is the glTF **directional light**, not `sceneData.sunlightDirection` — only the disabled
forward path reads that. `applySunDirection()` rewrites the light's node basis after the scene
graph refills `DrawContext::lights`, so shading, cascades and culling all see one direction.

Startup defaults are the tuned night look (sun at 0.02), so the engine boots into the state the
lamp and point-shadow work is measured against. Daylight is one dial.

Reverse-Z everywhere: `glm::perspective` is called with near/far swapped, depth clears to 0 and
compares `GREATER_OR_EQUAL`.

DDGI is implemented but parked — `DDGIFeature`, `AccelStructure` and `disabled_shaders/ddgi` are
out of the build.

## Performance

| Change | Effect |
|---|---|
| Tiled light culling — shade only the lights binned to a pixel's tile, not all 97 | Composite and transparent passes stop scaling with scene light count |
| Per-face cube-shadow culling — a caster inside a lamp's reach still only lands in 1–2 of 6 faces | ~6× fewer shadow draws |
| Offscreen light rejection — lamps whose whole 2–8 unit reach is off screen get no shadow map | Most of the 96 lamps skipped per frame |
| Shadow-slot hysteresis — an incumbent must be beaten by 35% before eviction | Stops shadows popping while panning |
| Read light members individually instead of copying a 96-byte struct per light, plus a squared-distance cull | Transparent forward **9.0 → 1.69 ms** |
| Bloom bright pass emits only the *excess* over threshold, not the whole colour | Fixed emissives at 40 blowing out neighbours; intensity back to a sane 0.5 |
| Half-res bloom chain | Bloom is low-frequency; full res bought nothing |
| Dropped the 8× MSAA colour/depth targets, whose only consumer was the disabled forward path | **~354 MB** of VRAM never written |

Combined, point shadows went from **+6.0 ms GPU / +3.4 ms CPU** to affordable enough to enable by
default: **10.78 → 8.73 ms GPU, 5.05 → 3.54 ms CPU** at 12 casters.

### Correctness fixes that were costing frame time

- Cascade slice corners are built from the camera basis and FOV, not by unprojecting the NDC cube —
  the latter ties the fit to the 0.1/100000 near/far and made cascade 0's radius **6198** instead
  of **7.56**.
- `PipelineBuilder::set_shaders` no longer forces a fragment stage, so depth-only shadow pipelines
  have none.
- `BarrierMerger` infers the depth aspect from *any* depth layout, not just `DEPTH_ATTACHMENT` —
  sampled shadow maps sit in `DEPTH_READ_ONLY`.
- The transparent pass no longer tonemaps: it emitted ACES + gamma, then blended onto a target the
  composite had already tonemapped. It now outputs linear HDR like everything else.
- Vertex normals use the inverse-transpose of the model matrix; the raw matrix skewed them under
  non-uniform scale.
