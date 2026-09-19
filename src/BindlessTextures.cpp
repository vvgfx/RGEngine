#include "BindlessTextures.h"
#include "fmt/base.h"
#include "vk_engine.h"

BindlessTextures *BindlessTextures::instance = nullptr;

BindlessTextures &BindlessTextures::Instance()
{
    if (instance == nullptr)
    {
        instance = new BindlessTextures();
    }
    return *instance;
}

void BindlessTextures::init(VkDevice device, DeletionQueue &delQueue)
{
    // PARTIALLY_BOUND lets us declare MAX_TEXTURES up front and leave unused slots unwritten.
    // No UPDATE_AFTER_BIND: every texture registers during init, before this set is ever bound, and that flag
    // would additionally require descriptorBindingSampledImageUpdateAfterBind.
    VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                          .bindingCount = 1,
                                                          .pBindingFlags = &bindingFlags};

    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_TEXTURES);
    layout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, &flagsInfo);

    VkDescriptorPoolSize poolSize{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_TEXTURES};
    VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 1,
                                        .poolSizeCount = 1,
                                        .pPoolSizes = &poolSize};
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));

    VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout};
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyDescriptorPool(device, pool, nullptr);
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        });
}

uint32_t BindlessTextures::Register(VkDevice device, VkImageView view, VkSampler sampler)
{
    if (count >= MAX_TEXTURES)
    {
        fmt::println("BindlessTextures: table full ({} textures), reusing slot 0", MAX_TEXTURES);
        return 0;
    }

    const uint32_t index = count++;

    DescriptorWriter writer;
    writer.write_image(0, view, sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, index);
    writer.update_set(device, set);

    return index;
}
