#pragma once
#include "vk_types.h"
#include <vector>

struct DrawContext;
struct DeletionQueue;
class VulkanEngine;

/// Per-instance data a ray-query hit needs, indexed by gl_InstanceCustomIndexEXT.
struct GPUGeometryInfo
{
    VkDeviceAddress vertexAddress;
    VkDeviceAddress indexAddress;
    glm::vec4 colorFactor;
    uint32_t firstIndex;
    uint32_t albedoTexIndex;
    uint32_t _pad[2];
};

/**
 * @brief One BLAS per submesh plus a TLAS over them, built once from a DrawContext.
 *
 * Only opaque surfaces are included: transparent geometry should not occlude or bounce diffuse GI.
 */
class AccelStructure
{
  public:
    /// Returns false if the device lacks ray query, in which case nothing is allocated.
    bool Build(VulkanEngine *engine, VkDevice device, const DrawContext &ctx, DeletionQueue &delQueue);

    VkAccelerationStructureKHR GetTLAS() const
    {
        return tlas;
    }
    VkBuffer GetGeometryBuffer() const
    {
        return geometryBuffer.buffer;
    }
    size_t GetGeometryBufferSize() const
    {
        return geometryBufferSize;
    }
    bool IsValid() const
    {
        return tlas != VK_NULL_HANDLE;
    }

  private:
    struct Blas
    {
        VkAccelerationStructureKHR handle;
        AllocatedBuffer buffer;
        VkDeviceAddress address;
    };

    std::vector<Blas> blases;

    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    AllocatedBuffer tlasBuffer;
    AllocatedBuffer geometryBuffer;
    size_t geometryBufferSize = 0;
};
