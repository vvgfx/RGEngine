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
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
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
};

// push constants for our mesh object draws
struct GPUDrawPushConstants
{
    glm::mat4 modelMatrix;
    VkDeviceAddress vertexBuffer;
};

// One vertex of the debug line overlay (LINE_LIST). position.w unused; kept vec4 to avoid
// std430 alignment surprises with buffer_reference. Must match shaders/debug/line.vert.
struct DebugLineVertex
{
    glm::vec4 position;
    glm::vec4 color;
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
};