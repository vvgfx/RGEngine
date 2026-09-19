#include "MaterialSystem.h"
#include "BindlessTextures.h"

void MaterialSystem::build_descriptors(VkDevice device)
{

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // material data.
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // color texture.
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // metallic-roughness texture.
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // tangent-space normal map.

    materialLayout = layoutBuilder.build(device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
}

MaterialInstance MaterialSystem::write_material(VkDevice device, MaterialPass pass, const MaterialResources &resources,
                                                DescriptorAllocatorGrowable &descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    matData.colorFactor = resources.colorFactors;
    matData.albedoTexIndex = BindlessTextures::Instance().Register(device, resources.colorImage.imageView, resources.colorSampler);

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, resources.normalImage.imageView, resources.normalSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void MaterialSystem::clear_resources(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
}