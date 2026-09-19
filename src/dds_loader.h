#pragma once
#include "vk_types.h"
#include <filesystem>
#include <optional>

/**
 * @brief Loads a BC-compressed DDS straight to the GPU.
 *
 * Block-compressed formats are native to Vulkan, so the payload is uploaded verbatim: no decode, no
 * mip generation, and roughly 4x less VRAM than the RGBA8 path stb_image produces.
 */
std::optional<AllocatedImage> load_dds(const std::filesystem::path &path);
