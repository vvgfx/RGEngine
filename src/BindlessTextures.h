#pragma once
#include "vk_descriptors.h"
#include "vk_types.h"

struct DeletionQueue;

/**
 * @brief A global, append-only table of textures addressable from any shader by index.
 *
 * Ray queries have no hit shaders, so a probe-trace shader cannot bind a per-material descriptor set.
 * It instead reads an index out of the geometry table and samples through this one array.
 */
struct BindlessTextures
{
    static constexpr uint32_t MAX_TEXTURES = 1024;

    static BindlessTextures &Instance();

    void init(VkDevice device, DeletionQueue &delQueue);

    /// Appends a texture and returns the index shaders use to sample it. Safe to call after init.
    uint32_t Register(VkDevice device, VkImageView view, VkSampler sampler);

    VkDescriptorSetLayout GetLayout() const
    {
        return layout;
    }
    VkDescriptorSet GetSet() const
    {
        return set;
    }

  private:
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    uint32_t count = 0;

    static BindlessTextures *instance;
};
