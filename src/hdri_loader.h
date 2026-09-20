#pragma once
#include "vk_types.h"
#include <filesystem>
#include <optional>

/**
 * @brief Load an equirectangular HDR environment map.
 *
 * Radiance .hdr only, which stb_image already decodes -- no new dependency. glTF cannot carry this:
 * core glTF 2.0 has no HDR and no cubemaps, and the one extension that does (EXT_lights_image_based)
 * is unsupported by fastgltf, so the path is hardcoded the same way the scene's is.
 *
 * Stored as RGBA16F rather than the RGBA32F stb hands back: half the memory, and far more range
 * than a sky needs. A 4K equirect is 134 MB as 32F against 67 MB as 16F.
 */
struct LoadedHDRI
{
    AllocatedImage radiance;   ///< full-resolution map, for the background and reflections
    AllocatedImage irradiance; ///< small cosine-convolved map, for the diffuse ambient term
};

/**
 * The irradiance map is not optional. Sampling a high mip of the radiance map as "ambient" is
 * wrong by orders of magnitude: a mip is a small-angle box blur, while diffuse irradiance is a
 * cosine-weighted integral over the whole hemisphere. An HDRI sun at 10,000 survives into mip 6
 * and blasts any surface whose normal happens to point at it.
 *
 * Convolution is done by projecting onto 9 spherical harmonic coefficients and evaluating them
 * back out -- the same representation EXT_lights_image_based standardises. SH9 captures diffuse
 * irradiance to within a few percent because the cosine lobe is almost entirely band-limited to
 * the first three bands.
 */
std::optional<LoadedHDRI> load_hdri(const std::filesystem::path &path);
