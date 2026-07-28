#include "DebugDrawFeature.h"
#include "rgraph/Rendergraph.h"
#include "vk_descriptors.h"
#include "vk_pipelines.h"
#include <algorithm>
#include <cstring>
#include <vulkan/vulkan_core.h>

using namespace rgraph;

DebugDrawFeature::DebugDrawFeature(VkDevice device, GPUSceneData &scene, VkDescriptorSetLayout gpuSceneLayout, VkFormat colorFormat,
                                   VkFormat depthFormat, DeletionQueue &delQueue)
    : sceneData(scene)
{
    _gpuSceneDataDescriptorLayout = gpuSceneLayout;

    VkShaderModule vert{}, frag{};
    if (!vkutil::load_shader_module("../shaders/debug/line.vert.spv", device, &vert))
        fmt::println("DebugDrawFeature: failed to load line.vert.spv");
    if (!vkutil::load_shader_module("../shaders/debug/line.frag.spv", device, &frag))
        fmt::println("DebugDrawFeature: failed to load line.frag.spv");

    // push constant carries the device address of the per-frame line vertex buffer
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset = 0;
    pc.size = sizeof(VkDeviceAddress);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &_gpuSceneDataDescriptorLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipeline.layout));

    PipelineBuilder builder;
    builder.set_shaders(vert, frag);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    builder.set_multisampling_none();
    builder.set_color_attachment_format(colorFormat);
    builder.disable_blending();
    builder.set_depth_format(depthFormat);
    builder.disable_depthtest(); // overlay draws on top of the meshes
    builder._pipelineLayout = pipeline.layout;
    pipeline.pipeline = builder.build_pipeline(device);

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        });
}

void DebugDrawFeature::Register(Rendergraph *builder)
{
    // Register() runs every frame during Build(); skip entirely when off or empty.
    if (!enabled || lineVerts.empty())
        return;

    builder->AddGraphicsPass(
        "Debug Overlay",
        [this](Pass &pass)
        {
            // clear=nullptr -> loadOp=LOAD: preserve the composited scene and draw over it
            pass.AddColorAttachment("drawImage", false, nullptr);
            pass.AddDepthStencilAttachment("depth_gbuf", false, nullptr);
            pass.CreatesBuffer("debugSceneBuffer", sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            pass.CreatesBuffer("debugLineBuffer", MAX_VERTS * sizeof(DebugLineVertex),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        },
        [this](PassExecution &passExec) { draw(passExec); });
}

void DebugDrawFeature::draw(PassExecution &passExec)
{
    AllocatedBuffer sceneBuf = passExec.allocatedBuffers["debugSceneBuffer"];
    *(GPUSceneData *)sceneBuf.info.pMappedData = sceneData;
    VkDescriptorSet set0 = passExec.frameDescriptor->allocate(passExec._device, _gpuSceneDataDescriptorLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, sceneBuf.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(passExec._device, set0);

    AllocatedBuffer lineBuf = passExec.allocatedBuffers["debugLineBuffer"];
    std::size_t count = std::min(lineVerts.size(), MAX_VERTS);
    std::memcpy(lineBuf.info.pMappedData, lineVerts.data(), count * sizeof(DebugLineVertex));

    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = lineBuf.buffer;
    VkDeviceAddress addr = vkGetBufferDeviceAddress(passExec._device, &addrInfo);

    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set0, 0, nullptr);

    VkViewport viewport{};
    viewport.width = (float)passExec._drawExtent.width;
    viewport.height = (float)passExec._drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(passExec.cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent.width = passExec._drawExtent.width;
    scissor.extent.height = passExec._drawExtent.height;
    vkCmdSetScissor(passExec.cmd, 0, 1, &scissor);

    vkCmdPushConstants(passExec.cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VkDeviceAddress), &addr);
    vkCmdDraw(passExec.cmd, (uint32_t)count, 1, 0, 0);

    passExec.drawCalls = 1;
    passExec.triangles = 0;
}
