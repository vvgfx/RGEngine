#include "hdri_loader.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include <cstring>
#include <stb_image.h>
#include <cmath>
#include <vector>

namespace
{
    /// IEEE 754 binary32 -> binary16. Sky values are well inside half range, so this only has to
    /// handle the normal case plus a clamp; denormals round to zero.
    uint16_t floatToHalf(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));

        const uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exponent = int32_t((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mantissa = x & 0x007FFFFFu;

        if (exponent >= 31) // overflow or inf/nan -> largest finite half
        {
            return uint16_t(sign | 0x7BFFu);
        }
        if (exponent <= 0) // underflow -> zero
        {
            return uint16_t(sign);
        }
        return uint16_t(sign | (uint32_t(exponent) << 10) | (mantissa >> 13));
    }

    constexpr uint32_t IRRADIANCE_W = 64;
    constexpr uint32_t IRRADIANCE_H = 32;

    /// Direction for an equirect texel. Must match skyEquirectUV() in sky.glsl exactly, or the
    /// irradiance map is rotated relative to the sky it was convolved from.
    void equirectDir(float u, float v, float &x, float &y, float &z)
    {
        const float phi = (u - 0.5f) * 2.0f * 3.14159265359f;
        const float theta = (v - 0.5f) * 3.14159265359f; // elevation, not polar angle
        const float c = std::cos(theta);
        x = c * std::cos(phi);
        y = std::sin(theta);
        z = c * std::sin(phi);
    }
} // namespace

std::optional<LoadedHDRI> load_hdri(const std::filesystem::path &path)
{
    if (!std::filesystem::exists(path))
    {
        fmt::println("HDRI: not found: {}", path.string());
        return {};
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    // Force 4 channels: Vulkan has no widely-supported 3-component float sampled format.
    float *pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, 4);
    if (pixels == nullptr)
    {
        fmt::println("HDRI: stbi_loadf failed for {} ({})", path.string(), stbi_failure_reason());
        return {};
    }

    const size_t texels = size_t(width) * size_t(height);
    std::vector<uint16_t> half(texels * 4);
    for (size_t i = 0; i < texels * 4; i++)
    {
        half[i] = floatToHalf(pixels[i]);
    }

    const VkExtent3D extent{uint32_t(width), uint32_t(height), 1};
    // Mipmapped so reflections off rough surfaces can pick a blurrier level.
    AllocatedImage radiance = GPUResourceAllocator::Instance().create_image(
        half.data(), extent, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, true, 8);

    // ---- cosine-convolve into a tiny irradiance map ----
    // Brute force over a downsampled copy rather than a spherical-harmonic projection: same result
    // for diffuse, and nothing subtle to get wrong. For each output direction, integrate every
    // input direction weighted by cos(angle) and solid angle. 512 x 128 taps costs nothing.
    constexpr uint32_t SRC_W = 32, SRC_H = 16;

    std::vector<float> small(SRC_W * SRC_H * 4, 0.0f); // rgb + count
    for (int y = 0; y < height; y++)
    {
        const uint32_t by = uint32_t(y) * SRC_H / height;
        for (int x = 0; x < width; x++)
        {
            const size_t o = (size_t(y) * width + x) * 4;
            const size_t b = (size_t(by) * SRC_W + uint32_t(x) * SRC_W / width) * 4;
            small[b + 0] += pixels[o + 0];
            small[b + 1] += pixels[o + 1];
            small[b + 2] += pixels[o + 2];
            small[b + 3] += 1.0f;
        }
    }
    stbi_image_free(pixels);

    std::vector<uint16_t> irr(size_t(IRRADIANCE_W) * IRRADIANCE_H * 4);
    for (uint32_t oy = 0; oy < IRRADIANCE_H; oy++)
    {
        for (uint32_t ox = 0; ox < IRRADIANCE_W; ox++)
        {
            float nx, ny, nz;
            equirectDir((ox + 0.5f) / IRRADIANCE_W, (oy + 0.5f) / IRRADIANCE_H, nx, ny, nz);

            double sum[3] = {};
            for (uint32_t sy = 0; sy < SRC_H; sy++)
            {
                for (uint32_t sx = 0; sx < SRC_W; sx++)
                {
                    float dx, dy, dz;
                    equirectDir((sx + 0.5f) / SRC_W, (sy + 0.5f) / SRC_H, dx, dy, dz);
                    const float cosTheta = nx * dx + ny * dy + nz * dz;
                    if (cosTheta <= 0.0f)
                    {
                        continue; // below the hemisphere
                    }
                    // Solid angle is uniform in longitude, cosine-weighted in elevation.
                    const size_t b = (size_t(sy) * SRC_W + sx) * 4;
                    const double w = double(cosTheta) * std::sqrt(std::max(0.0f, 1.0f - dy * dy)) *
                                     (2.0 * 3.14159265358979 / SRC_W) * (3.14159265358979 / SRC_H) / std::max(1.0f, small[b + 3]);
                    for (int c = 0; c < 3; c++)
                    {
                        sum[c] += small[b + c] * w;
                    }
                }
            }

            const size_t o = (size_t(oy) * IRRADIANCE_W + ox) * 4;
            // /pi turns irradiance into the outgoing radiance of a Lambertian surface, which is
            // what the shader multiplies by albedo.
            for (int c = 0; c < 3; c++)
            {
                irr[o + c] = floatToHalf(float(sum[c] / 3.14159265358979));
            }
            irr[o + 3] = floatToHalf(1.0f);
        }
    }

    AllocatedImage irradiance = GPUResourceAllocator::Instance().create_image(
        irr.data(), VkExtent3D{IRRADIANCE_W, IRRADIANCE_H, 1}, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false, 8);

    fmt::println("HDRI: {}x{} from {} ({:.0f} MB) + {}x{} SH9 irradiance", width, height, path.filename().string(),
                 double(half.size() * sizeof(uint16_t)) / 1048576.0, IRRADIANCE_W, IRRADIANCE_H);

    return LoadedHDRI{radiance, irradiance};
}
