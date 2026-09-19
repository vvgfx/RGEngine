#include "dds_loader.h"
#include "GPUResourceAllocator.h"
#include "fmt/base.h"
#include "vk_engine.h"
#include "vk_images.h"
#include <cstring>
#include <fstream>

namespace
{
    // DDS: 4-byte magic, then a 124-byte DDS_HEADER, then (when fourCC is "DX10") a 20-byte DDS_HEADER_DXT10.
    constexpr size_t DDS_HEADER_SIZE = 4 + 124;
    constexpr size_t DDS_DX10_SIZE = 20;

    uint32_t readU32(const std::vector<uint8_t> &b, size_t offset)
    {
        uint32_t v;
        std::memcpy(&v, b.data() + offset, sizeof(v));
        return v;
    }
} // namespace

std::optional<AllocatedImage> load_dds(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return {};
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    if (bytes.size() < DDS_HEADER_SIZE + DDS_DX10_SIZE || std::memcmp(bytes.data(), "DDS ", 4) != 0)
    {
        return {};
    }

    const uint32_t height = readU32(bytes, 12);
    const uint32_t width = readU32(bytes, 16);
    uint32_t mipLevels = readU32(bytes, 28);
    if (mipLevels == 0)
    {
        mipLevels = 1;
    }

    if (std::memcmp(bytes.data() + 84, "DX10", 4) != 0)
    {
        fmt::println("DDS: {} is not DX10, unsupported", path.string());
        return {};
    }

    const uint32_t dxgiFormat = readU32(bytes, DDS_HEADER_SIZE);

    VkFormat format;
    uint32_t blockBytes;
    switch (dxgiFormat)
    {
    // Deliberately UNORM even for the SRGB variant: the G-buffer shaders already linearise with
    // pow(2.2), matching the stb_image path. Letting the hardware do it too would double-darken.
    case 98: // DXGI_FORMAT_BC7_UNORM
    case 99: // DXGI_FORMAT_BC7_UNORM_SRGB
        format = VK_FORMAT_BC7_UNORM_BLOCK;
        blockBytes = 16;
        break;
    case 71: // BC1_UNORM
    case 72: // BC1_UNORM_SRGB
        format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        blockBytes = 8;
        break;
    case 77: // BC3_UNORM
    case 78: // BC3_UNORM_SRGB
        format = VK_FORMAT_BC3_UNORM_BLOCK;
        blockBytes = 16;
        break;
    case 83: // BC5_UNORM
        format = VK_FORMAT_BC5_UNORM_BLOCK;
        blockBytes = 16;
        break;
    default:
        fmt::println("DDS: {} has unsupported DXGI format {}", path.string(), dxgiFormat);
        return {};
    }

    const size_t dataOffset = DDS_HEADER_SIZE + DDS_DX10_SIZE;

    // Walk the mip chain, clamping to the declared payload so a truncated file cannot overrun.
    std::vector<VkBufferImageCopy> regions;
    size_t cursor = dataOffset;
    uint32_t usedMips = 0;

    for (uint32_t level = 0; level < mipLevels; level++)
    {
        const uint32_t w = std::max(1u, width >> level);
        const uint32_t h = std::max(1u, height >> level);
        const size_t size = size_t((w + 3) / 4) * ((h + 3) / 4) * blockBytes;

        if (cursor + size > bytes.size())
        {
            break;
        }

        regions.push_back(VkBufferImageCopy{.bufferOffset = cursor - dataOffset,
                                            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                                            .imageExtent = {w, h, 1}});
        cursor += size;
        usedMips++;
    }

    if (usedMips == 0)
    {
        return {};
    }

    auto &alloc = GPUResourceAllocator::Instance();
    VulkanEngine &engine = VulkanEngine::Instance();

    const size_t payload = cursor - dataOffset;
    AllocatedBuffer staging = alloc.create_buffer(payload, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    std::memcpy(staging.info.pMappedData, bytes.data() + dataOffset, payload);

    // vkinit::image_create_info hardcodes one mip level, so build the create info by hand.
    VkImageCreateInfo imgInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = format,
                              .extent = {width, height, 1},
                              .mipLevels = usedMips,
                              .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};

    VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_GPU_ONLY,
                                      .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

    AllocatedImage image{};
    image.imageFormat = format;
    image.imageExtent = {width, height, 1};

    alloc.create_image(&imgInfo, &allocInfo, &image.image, &image.allocation, nullptr);
    if (image.image == VK_NULL_HANDLE)
    {
        alloc.destroy_buffer(staging);
        fmt::println("DDS: allocation failed for {}", path.string());
        return {};
    }

    engine.immediate_submit(
        [&](VkCommandBuffer cmd)
        {
            VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = usedMips, .layerCount = 1};

            VkImageMemoryBarrier2 toDst{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        .image = image.image,
                                        .subresourceRange = range};
            VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDst};
            vkCmdPipelineBarrier2(cmd, &dep);

            vkCmdCopyBufferToImage(cmd, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());

            VkImageMemoryBarrier2 toRead = toDst;
            toRead.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toRead.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toRead.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            toRead.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            dep.pImageMemoryBarriers = &toRead;
            vkCmdPipelineBarrier2(cmd, &dep);
        });

    alloc.destroy_buffer(staging);

    VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .image = image.image,
                                   .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = format,
                                   .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, usedMips, 0, 1}};
    VK_CHECK(vkCreateImageView(engine.GetVkDevice(), &viewInfo, nullptr, &image.imageView));

    return image;
}
