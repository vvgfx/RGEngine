#pragma once
#include <span>
#include <vk_mem_alloc.h>
#include <vk_types.h>

class VulkanEngine;

class GPUResourceAllocator
{
  public:
    static void init(VmaAllocator &_allocator, VkDevice _device);

    static GPUResourceAllocator &Instance();

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    void create_image(VkImageCreateInfo *pImageCreateInfo, VmaAllocationCreateInfo *pAllocationCreateInfo, VkImage *pImage,
                      VmaAllocation *pAllocation, VmaAllocationInfo *pAllocationInfo);

    void destroy_image(VkImage image, VmaAllocation allocation);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    void destroy_image(const AllocatedImage &img);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);

    void cleanup();

    VkDevice getDevice();
    VkPhysicalDevice getPhysicalDevice() const;
    VmaAllocator getAllocator() const;

  private:
    VmaAllocator _allocator;
    VkDevice _device;
    VulkanEngine *_engine;
    static GPUResourceAllocator *instance;
};