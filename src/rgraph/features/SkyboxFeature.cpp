#include "rgraph/features/SkyboxFeature.h"
#include "vk_descriptors.h"
#include "vk_engine.h"
#include "vk_pipelines.h"
#include "vk_types.h"
#include <vulkan/vulkan_core.h>


rgraph::SkyboxFeature::SkyboxFeature(VkDevice _device, DeletionQueue &delQueue)
{
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
}