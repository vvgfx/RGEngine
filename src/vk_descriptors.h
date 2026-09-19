#pragma once

#include <deque>
#include <span>
#include <vk_types.h>

struct DescriptorLayoutBuilder
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    // count > 1 declares a descriptor array (bindless texture tables).
    void add_binding(uint32_t binding, VkDescriptorType type, uint32_t count = 1);
    void clear();
    VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages, void *pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0);
};

struct DescriptorAllocator
{
    struct PoolSizeRatio
    {
        VkDescriptorType type;
        float ratio;
    };

    VkDescriptorPool pool;

    void init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clear_descriptors(VkDevice device);
    void destroy_pool(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};

struct DescriptorAllocatorGrowable
{
  public:
    struct PoolSizeRatio
    {
        VkDescriptorType type;
        float ratio;
    };

    void init(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios, VkDescriptorPoolCreateFlags flags = 0);
    void clear_pools(VkDevice device);
    void destroy_pools(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout, void *pNext = nullptr);

  private:
    VkDescriptorPool get_pool(VkDevice device);
    VkDescriptorPool create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios);

    VkDescriptorPoolCreateFlags poolFlags = 0;

    std::vector<PoolSizeRatio> ratios;
    std::vector<VkDescriptorPool> fullPools;
    std::vector<VkDescriptorPool> readyPools;
    uint32_t setsPerPool;
};

struct DescriptorWriter
{
    std::deque<VkDescriptorImageInfo> imageInfos;
    std::deque<VkDescriptorBufferInfo> bufferInfos;
    std::deque<VkWriteDescriptorSetAccelerationStructureKHR> accelInfos;
    std::vector<VkWriteDescriptorSet> writes;

    void write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type, uint32_t arrayElement = 0);
    void write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type);

    /// A TLAS descriptor carries its handle in pNext rather than in the write itself.
    void write_accel(int binding, const VkAccelerationStructureKHR *accel);

    void clear();
    void update_set(VkDevice device, VkDescriptorSet set);
};
