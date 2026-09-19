#include "LightDebugFeature.h"
#include "fmt/base.h"
#include "vk_pipelines.h"
#include <algorithm>

namespace
{
    // Must match SEG in light_sphere.vert: three rings, two vertices per segment.
    constexpr uint32_t SEGMENTS = 48;
    constexpr uint32_t VERTS_PER_SPHERE = 3 * SEGMENTS * 2;
} // namespace

rgraph::LightDebugFeature::LightDebugFeature(VkDevice device, DeletionQueue &delQueue, DrawContext &drawContext, GPUSceneData &sceneData,
                                             VkFormat colorFormat, VkFormat depthFormat)
    : drawContext(drawContext), sceneData(sceneData)
{
    VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(SpherePush)};
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipeline.layout));

    VkShaderModule vert;
    VkShaderModule frag;
    if (!vkutil::load_shader_module("../shaders/debug/light_sphere.vert.spv", device, &vert) ||
        !vkutil::load_shader_module("../shaders/debug/light_sphere.frag.spv", device, &frag))
    {
        fmt::println("LightDebug: failed to load light_sphere shaders");
    }
    else
    {
        PipelineBuilder builder;
        builder.set_shaders(vert, frag);
        builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        builder.set_polygon_mode(VK_POLYGON_MODE_FILL); // ignored for lines
        builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        builder.set_multisampling_none();
        builder.disable_blending();
        builder.set_color_attachment_format(colorFormat);
        builder.set_depth_format(depthFormat);

        // Test against the scene depth so spheres are correctly hidden behind geometry -- that
        // occlusion is the whole point, it is what shows a lamp sitting inside a wall. No depth
        // write: the overlay must not disturb anything sampling depth later.
        builder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL); // reverse-Z
        builder._pipelineLayout = pipeline.layout;

        pipeline.pipeline = builder.build_pipeline(device);
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    delQueue.push_function(
        [device, this]()
        {
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
        });
}

void rgraph::LightDebugFeature::Register(rgraph::Rendergraph *builder)
{
    if (!settings.enabled || pipeline.pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    builder->AddGraphicsPass(
        "light-debug",
        [](Pass &pass)
        {
            // Load, never clear: this draws on top of the already-composited scene.
            pass.AddColorAttachment("drawImage", false, nullptr);
            pass.AddDepthStencilAttachment("depth_gbuf", true, nullptr);
        },
        [&](PassExecution &passExec) { renderPass(passExec); });
}

void rgraph::LightDebugFeature::renderPass(rgraph::PassExecution &passExec)
{
    vkCmdBindPipeline(passExec.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

    uint32_t draws = 0;

    for (const GPULightingData &light : drawContext.lights)
    {
        if (light.type == 0 && !settings.showDirectional)
        {
            continue;
        }

        SpherePush push{};
        push.viewproj = sceneData.viewproj;

        // A directional light has no position or reach, so mark it with a fixed-size sphere at the
        // node's origin just to show where the exporter put it.
        const float radius = light.type == 0 ? 1.0f : std::max(light.range, 0.05f) * settings.radiusScale;
        push.posRadius = glm::vec4(glm::vec3(light.transform[3]), radius);
        push.color = glm::vec4(light.color, 1.0f);

        vkCmdPushConstants(passExec.cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SpherePush), &push);
        vkCmdDraw(passExec.cmd, VERTS_PER_SPHERE, 1, 0, 0);
        draws++;
    }

    passExec.drawCalls = float(draws);
    passExec.triangles = 0;
}
