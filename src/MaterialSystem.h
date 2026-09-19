#pragma once
#include "vk_descriptors.h"
#include "vk_types.h"

struct MaterialSystemCreateInfo
{
    VkDevice _device;
    VkFormat colorFormat, depthFormat;

    // TODO : come back and fix this ugly line.
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
};

// GLTF Metalllic-Roughness material system.
struct MaterialSystem
{
    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants
    {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // padding, we need it anyway for uniform buffers
        glm::vec4 extra[14];
    };

    // All this data is written into a descriptor set. It is not stored on the CPU after loading.
    struct MaterialResources
    {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        AllocatedImage normalImage;
        VkSampler normalSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
        glm::vec4 colorFactors{1.0f};
    };

    DescriptorWriter writer;

    void build_descriptors(VkDevice device);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources &resources,
                                    DescriptorAllocatorGrowable &descriptorAllocator);
};