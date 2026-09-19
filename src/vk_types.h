// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#define VK_CHECK(x)                                                                                                                                  \
    do                                                                                                                                               \
    {                                                                                                                                                \
        VkResult err = x;                                                                                                                            \
        if (err)                                                                                                                                     \
        {                                                                                                                                            \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err));                                                                         \
            abort();                                                                                                                                 \
        }                                                                                                                                            \
    } while (0)

// Represents a single image in the GPU
struct AllocatedImage
{
    // default-initialised so callers can test for "unset" rather than reading indeterminate handles
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkExtent3D imageExtent{};
    VkFormat imageFormat = VK_FORMAT_UNDEFINED;
};

// represents a single buffer in the GPU
struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

// This is the per-vertex data structure on the CPU side. This must match whatever is on the GPU side (shaders)
struct Vertex
{
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// holds the GPU resources of a mesh (Vertex buffer, Index Buffer and Vertex Buffer address)
// To visualize, each of this will be the GPU-equivalent of one unique object in a blender scene.
struct GPUMeshBuffers
{
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;

    // This holds the address of the vertex buffer. This is the same as RenderObject::vertexBufferAddress and is used in
    // GPUPushConstants. We require the address of the vertex buffer because we directly reference it as a pointer in
    // the shader via push constants.
    VkDeviceAddress vertexBufferAddress;

    // needed by ray-query hit shading to pull indices via buffer_reference.
    VkDeviceAddress indexBufferAddress;

    // BLAS builds need the vertex count as a bound.
    uint32_t vertexCount = 0;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants
{
    glm::mat4 modelMatrix;
    VkDeviceAddress vertexBuffer;
};

enum class MaterialPass : uint8_t
{
    MainColor,
    Transparent,
    Other
};
struct MaterialPipeline
{
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct MaterialInstance
{
    VkDescriptorSet materialSet;
    MaterialPass passType;

    // Ray-query hit shading cannot bind materialSet, so it reads these out of the geometry table instead.
    uint32_t albedoTexIndex = 0;
    glm::vec4 colorFactor{1.0f};
};