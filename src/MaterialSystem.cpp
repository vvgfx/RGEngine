#include "MaterialSystem.h"
#include "BindlessTextures.h"
#include "vk_engine.h"

void MaterialSystem::build_descriptors(VkDevice device)
{

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // material data.
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // color texture.
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // metallic-roughness texture.
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // tangent-space normal map.
    layoutBuilder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // emissive map.

    materialLayout = layoutBuilder.build(device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
}

MaterialInstance MaterialSystem::write_material(VkDevice device, MaterialPass pass, const MaterialResources &resources,
                                                DescriptorAllocatorGrowable &descriptorAllocator)
{
    VulkanEngine &engine = VulkanEngine::Instance();

    // A material without a normal or emissive map still has to write a valid descriptor; writing a
    // null image view is undefined behaviour and crashed init_default_data.
    auto view = [&](const AllocatedImage &img, const AllocatedImage &fallback)
    { return img.imageView != VK_NULL_HANDLE ? img.imageView : fallback.imageView; };
    auto sampler = [&](VkSampler s) { return s != VK_NULL_HANDLE ? s : engine.GetDefaultSampler(); };

    const VkImageView colorView = view(resources.colorImage, engine.GetDefaultImage());
    const VkImageView metalRoughView = view(resources.metalRoughImage, engine.GetDefaultImage());
    const VkImageView normalView = view(resources.normalImage, engine.GetFlatNormalImage());
    const VkImageView emissiveView = view(resources.emissiveImage, engine.GetDefaultImage());

    MaterialInstance matData;
    matData.passType = pass;
    matData.colorFactor = resources.colorFactors;
    matData.albedoTexIndex = BindlessTextures::Instance().Register(device, colorView, sampler(resources.colorSampler));

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, colorView, sampler(resources.colorSampler), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, metalRoughView, sampler(resources.metalRoughSampler), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, normalView, sampler(resources.normalSampler), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(4, emissiveView, sampler(resources.emissiveSampler), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void MaterialSystem::clear_resources(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
}