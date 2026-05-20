#include "SMAAFeature.h"
#include "GPUResourceAllocator.h"
#include "rgraph/Rendergraph.h"
#include "vk_descriptors.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include <vulkan/vulkan_core.h>

// These headers contain the raw byte arrays for the SMAA area and search LUTs.
// Include in exactly one .cpp file.
#include "smaa/AreaTex.h"
#include "smaa/SearchTex.h"

rgraph::SMAAFeature::SMAAFeature(VkDevice device, VkExtent3D extent, DeletionQueue &delQueue)
    : _device(device), _extent(extent)
{
    createLUTs(delQueue);
    createImages(delQueue);
    createPipelines();

    delQueue.push_function(
        [device, this]()
        {
            vkDestroySampler(device, linearSampler, nullptr);
            vkDestroyDescriptorSetLayout(device, edgeDescLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, blendDescLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, neighborDescLayout, nullptr);
            vkDestroyPipelineLayout(device, edgePipeline.layout, nullptr);
            vkDestroyPipelineLayout(device, blendPipeline.layout, nullptr);
            vkDestroyPipelineLayout(device, neighborPipeline.layout, nullptr);
            vkDestroyPipeline(device, edgePipeline.pipeline, nullptr);
            vkDestroyPipeline(device, blendPipeline.pipeline, nullptr);
            vkDestroyPipeline(device, neighborPipeline.pipeline, nullptr);
        });
}

void rgraph::SMAAFeature::createLUTs(DeletionQueue &delQueue)
{
    GPUResourceAllocator &gpuAlloc = GPUResourceAllocator::Instance();
    VulkanEngine &engine = VulkanEngine::Instance();

    auto uploadTexture = [&](const unsigned char *data, size_t dataSize, VkExtent3D extent, VkFormat format) -> AllocatedImage
    {
        AllocatedBuffer staging =
            gpuAlloc.create_buffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        memcpy(staging.info.pMappedData, data, dataSize);

        AllocatedImage img =
            gpuAlloc.create_image(extent, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        engine.immediate_submit(
            [&](VkCommandBuffer cmd)
            {
                vkutil::transition_image(cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                VkBufferImageCopy copy{};
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy.imageSubresource.layerCount = 1;
                copy.imageExtent = extent;
                vkCmdCopyBufferToImage(cmd, staging.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                       &copy);

                vkutil::transition_image(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            });

        gpuAlloc.destroy_buffer(staging);
        return img;
    };

    areaTexture = uploadTexture(areaTexBytes, AREATEX_SIZE, {AREATEX_WIDTH, AREATEX_HEIGHT, 1}, VK_FORMAT_R8G8_UNORM);
    searchTexture =
        uploadTexture(searchTexBytes, SEARCHTEX_SIZE, {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1}, VK_FORMAT_R8_UNORM);

    delQueue.push_function(
        [this]()
        {
            GPUResourceAllocator::Instance().destroy_image(areaTexture);
            GPUResourceAllocator::Instance().destroy_image(searchTexture);
        });
}

void rgraph::SMAAFeature::createImages(DeletionQueue &delQueue)
{
    GPUResourceAllocator &gpuAlloc = GPUResourceAllocator::Instance();

    constexpr VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    constexpr VkFormat kEdgesFormat = VK_FORMAT_R8G8B8A8_UNORM;

    smaaEdges = gpuAlloc.create_image(_extent, kEdgesFormat, colorUsage);
    smaaBlend = gpuAlloc.create_image(_extent, kEdgesFormat, colorUsage);

    Rendergraph::Instance().AddTrackedImage("smaaEdges", VK_IMAGE_LAYOUT_UNDEFINED, smaaEdges);
    Rendergraph::Instance().AddTrackedImage("smaaBlend", VK_IMAGE_LAYOUT_UNDEFINED, smaaBlend);

    delQueue.push_function(
        [this]()
        {
            GPUResourceAllocator &alloc = GPUResourceAllocator::Instance();
            alloc.destroy_image(smaaEdges);
            alloc.destroy_image(smaaBlend);
        });
}

void rgraph::SMAAFeature::createPipelines()
{
    VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(_device, &samplerInfo, nullptr, &linearSampler);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(glm::vec4);

    auto buildLayout = [&](std::initializer_list<VkDescriptorType> bindings) -> VkDescriptorSetLayout
    {
        DescriptorLayoutBuilder lb;
        uint32_t i = 0;
        for (VkDescriptorType type : bindings)
            lb.add_binding(i++, type);
        return lb.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    };

    edgeDescLayout    = buildLayout({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
    blendDescLayout   = buildLayout({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
    neighborDescLayout = buildLayout({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});

    auto buildPipelineLayout = [&](VkDescriptorSetLayout descLayout) -> VkPipelineLayout
    {
        VkPipelineLayoutCreateInfo info = vkinit::pipeline_layout_create_info();
        info.setLayoutCount = 1;
        info.pSetLayouts = &descLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &pcRange;
        VkPipelineLayout layout;
        VK_CHECK(vkCreatePipelineLayout(_device, &info, nullptr, &layout));
        return layout;
    };

    edgePipeline.layout    = buildPipelineLayout(edgeDescLayout);
    blendPipeline.layout   = buildPipelineLayout(blendDescLayout);
    neighborPipeline.layout = buildPipelineLayout(neighborDescLayout);

    auto loadShader = [&](const char *path) -> VkShaderModule
    {
        VkShaderModule mod;
        if (!vkutil::load_shader_module(path, _device, &mod))
            fmt::println("SMAA: failed to load shader {}", path);
        return mod;
    };

    auto buildPipeline = [&](VkShaderModule vert, VkShaderModule frag, VkPipelineLayout layout,
                              VkFormat colorFmt) -> VkPipeline
    {
        PipelineBuilder pb;
        pb.set_shaders(vert, frag);
        pb.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pb.set_polygon_mode(VK_POLYGON_MODE_FILL);
        pb.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pb.set_multisampling_none();
        pb.set_color_attachment_format(colorFmt);
        pb.disable_blending();
        pb.disable_depthtest();
        pb._pipelineLayout = layout;
        return pb.build_pipeline(_device);
    };

    constexpr VkFormat kEdgesFormat  = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    {
        VkShaderModule vert = loadShader("../shaders/smaa/edge.vert.spv");
        VkShaderModule frag = loadShader("../shaders/smaa/edge.frag.spv");
        edgePipeline.pipeline = buildPipeline(vert, frag, edgePipeline.layout, kEdgesFormat);
        vkDestroyShaderModule(_device, vert, nullptr);
        vkDestroyShaderModule(_device, frag, nullptr);
    }
    {
        VkShaderModule vert = loadShader("../shaders/smaa/blendweight.vert.spv");
        VkShaderModule frag = loadShader("../shaders/smaa/blendweight.frag.spv");
        blendPipeline.pipeline = buildPipeline(vert, frag, blendPipeline.layout, kEdgesFormat);
        vkDestroyShaderModule(_device, vert, nullptr);
        vkDestroyShaderModule(_device, frag, nullptr);
    }
    {
        VkShaderModule vert = loadShader("../shaders/smaa/neighbor.vert.spv");
        VkShaderModule frag = loadShader("../shaders/smaa/neighbor.frag.spv");
        neighborPipeline.pipeline = buildPipeline(vert, frag, neighborPipeline.layout, kOutputFormat);
        vkDestroyShaderModule(_device, vert, nullptr);
        vkDestroyShaderModule(_device, frag, nullptr);
    }
}

void rgraph::SMAAFeature::Register(Rendergraph *builder)
{
    static VkClearValue zeroClear{};
    zeroClear.color = {0.0f, 0.0f, 0.0f, 0.0f};

    builder->AddGraphicsPass(
        "SMAA Edge Detection",
        [](Pass &pass)
        {
            pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.AddColorAttachment("smaaEdges", true, &zeroClear);
        },
        [&](PassExecution &passExec) { edgePass(passExec); });

    builder->AddGraphicsPass(
        "SMAA Blend Weight",
        [](Pass &pass)
        {
            pass.ReadsImage("smaaEdges", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.AddColorAttachment("smaaBlend", true, &zeroClear);
        },
        [&](PassExecution &passExec) { blendWeightPass(passExec); });

    builder->AddGraphicsPass(
        "SMAA Neighborhood Blend",
        [](Pass &pass)
        {
            pass.ReadsImage("drawImage", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.ReadsImage("smaaBlend", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            pass.AddColorAttachment("smaaOutput", false, nullptr);
        },
        [&](PassExecution &passExec) { neighborBlendPass(passExec); });
}

static void setFullscreenViewportScissor(VkCommandBuffer cmd, VkExtent3D extent)
{
    VkViewport viewport{.x      = 0.0f,
                        .y      = 0.0f,
                        .width  = (float)extent.width,
                        .height = (float)extent.height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{.offset = {0, 0}, .extent = {extent.width, extent.height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void rgraph::SMAAFeature::edgePass(PassExecution &passExec)
{
    VkDescriptorSet desc = passExec.frameDescriptor->allocate(passExec._device, edgeDescLayout);
    DescriptorWriter writer;
    writer.write_image(0, passExec.allocatedImages.at("drawImage").imageView, linearSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(passExec._device, desc);

    glm::vec4 rtMetrics = {1.0f / (float)passExec._drawExtent.width,
                           1.0f / (float)passExec._drawExtent.height,
                           (float)passExec._drawExtent.width,
                           (float)passExec._drawExtent.height};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline.layout, 0, 1, &desc, 0,
                            nullptr);
    vkCmdPushConstants(passExec.cmd, edgePipeline.layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4),
                       &rtMetrics);
    setFullscreenViewportScissor(passExec.cmd, passExec._drawExtent);
    vkCmdDraw(passExec.cmd, 3, 1, 0, 0);

    passExec.drawCalls++;
}

void rgraph::SMAAFeature::blendWeightPass(PassExecution &passExec)
{
    VkDescriptorSet desc = passExec.frameDescriptor->allocate(passExec._device, blendDescLayout);
    DescriptorWriter writer;
    writer.write_image(0, smaaEdges.imageView, linearSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, areaTexture.imageView, linearSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, searchTexture.imageView, linearSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(passExec._device, desc);

    glm::vec4 rtMetrics = {1.0f / (float)passExec._drawExtent.width,
                           1.0f / (float)passExec._drawExtent.height,
                           (float)passExec._drawExtent.width,
                           (float)passExec._drawExtent.height};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blendPipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blendPipeline.layout, 0, 1, &desc, 0,
                            nullptr);
    vkCmdPushConstants(passExec.cmd, blendPipeline.layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4),
                       &rtMetrics);
    setFullscreenViewportScissor(passExec.cmd, passExec._drawExtent);
    vkCmdDraw(passExec.cmd, 3, 1, 0, 0);

    passExec.drawCalls++;
}

void rgraph::SMAAFeature::neighborBlendPass(PassExecution &passExec)
{
    VkDescriptorSet desc = passExec.frameDescriptor->allocate(passExec._device, neighborDescLayout);
    DescriptorWriter writer;
    writer.write_image(0, passExec.allocatedImages.at("drawImage").imageView, linearSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, smaaBlend.imageView, linearSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(passExec._device, desc);

    glm::vec4 rtMetrics = {1.0f / (float)passExec._drawExtent.width,
                           1.0f / (float)passExec._drawExtent.height,
                           (float)passExec._drawExtent.width,
                           (float)passExec._drawExtent.height};

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, neighborPipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, neighborPipeline.layout, 0, 1, &desc,
                            0, nullptr);
    vkCmdPushConstants(passExec.cmd, neighborPipeline.layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4),
                       &rtMetrics);
    setFullscreenViewportScissor(passExec.cmd, passExec._drawExtent);
    vkCmdDraw(passExec.cmd, 3, 1, 0, 0);

    passExec.drawCalls++;
}
