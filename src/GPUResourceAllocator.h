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
    /// bytesPerTexel defaults to 4 for the usual RGBA8 upload; HDR formats must pass their own
    /// (RGBA16F is 8) or the staging copy is short and the image comes out as garbage.
    AllocatedImage create_image(void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false,
                                uint32_t bytesPerTexel = 4);
    void destroy_image(const AllocatedImage &img);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);

    void cleanup();

    VkDevice getDevice();

    // prevent copy
    GPUResourceAllocator(const GPUResourceAllocator &) = delete;
    GPUResourceAllocator &operator=(const GPUResourceAllocator &) = delete;

    // we now need the default constructor to be manually specified.
    GPUResourceAllocator() = default;

  private:
    VmaAllocator _allocator;
    VkDevice _device;
    VulkanEngine *_engine;
    static GPUResourceAllocator *instance;
};