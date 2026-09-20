#include "GPUResourceAllocator.h"
#include "MaterialSystem.h"
#include "imgui.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_loader.h"
#include "vk_types.h"
#include <RGEngine.h>
#include <chrono>
#include <cmath>
#include <glm/trigonometric.hpp>
#include <memory>
#include <vulkan/vulkan_core.h>

// Last on purpose: vGizmo3D.h pulls "using namespace glm;" into the global namespace (it needs
// it for its own vec3 declarations), so including it here keeps that out of every header above.
#include "imGuIZMOquat.h"

void RGEngine::init()
{

    VulkanEngine::init();

    std::string structurePath = {"../assets/bistro_glb/bistro.glb"};

    // this is called after the pipelines are initialzed.
    auto structureFile = loadGltf(structurePath);

    assert(structureFile.has_value());

    loadedScenes["scene"] = *structureFile;

    structureFile.value()->name = "scene";

    rgraphInstance.Init(_device, _drawImage.imageExtent, _instance);

    VkExtent3D extent = {_windowExtent.width, _windowExtent.height, 1};
    computeFeature = std::make_shared<rgraph::ComputeBackgroundFeature>(_device, _mainDeletionQueue, extent, _drawImage);
    MaterialSystemCreateInfo msCreateInfo = {_device, _drawImage.imageFormat, _depthImage.imageFormat, _gpuSceneDataDescriptorLayout};
    // The legacy forward renderer. Never registered (see AddFeature below), but its constructor still
    // built pipelines, and it shares light_mesh.frag with the transparent pass - which now needs a
    // fourth set for the light grid that this one has no layout for.
    // PBRFeature = std::make_shared<rgraph::PBRShadingFeature>(mainDrawContext, _device, msCreateInfo, sceneData, _gpuSceneDataDescriptorLayout,
    //                                                          _mainDeletionQueue);

    // must precede the deferred feature: its composite pipeline layout needs the shadow set layout.
    shadowFeature = std::make_shared<rgraph::ShadowFeature>(_device, _mainDeletionQueue, mainDrawContext, sceneData);
    localShadowFeature = std::make_shared<rgraph::LocalShadowFeature>(_device, _mainDeletionQueue, mainDrawContext, sceneData);
    lightCullFeature =
        std::make_shared<rgraph::LightCullFeature>(_device, _mainDeletionQueue, mainDrawContext, sceneData, _drawImage.imageExtent);

    deferredFeature = std::make_shared<rgraph::DeferredRenderingFeature>(mainDrawContext, _device, sceneData, _gpuSceneDataDescriptorLayout,
                                                                         msCreateInfo, _mainDeletionQueue, shadowFeature.get(),
                                                                         localShadowFeature.get(), lightCullFeature.get());
    // create MSAA images. TODO: move these out somewhere later.
    // createMsaaImages(); // 8x MSAA targets, ~354MB, only used by the disabled PBRShadingFeature

    postImage = GPUResourceAllocator::Instance().create_image(
        _drawImage.imageExtent, _drawImage.imageFormat,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    _mainDeletionQueue.push_function([this]() { GPUResourceAllocator::Instance().destroy_image(postImage); });

    // Half res: bloom is a wide low-frequency effect, so full res buys nothing but cost.
    const VkExtent3D bloomExtent{_drawImage.imageExtent.width / 2, _drawImage.imageExtent.height / 2, 1};
    const VkImageUsageFlags bloomUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    bloomA = GPUResourceAllocator::Instance().create_image(bloomExtent, _drawImage.imageFormat, bloomUsage);
    bloomB = GPUResourceAllocator::Instance().create_image(bloomExtent, _drawImage.imageFormat, bloomUsage);
    _mainDeletionQueue.push_function(
        [this]()
        {
            GPUResourceAllocator::Instance().destroy_image(bloomA);
            GPUResourceAllocator::Instance().destroy_image(bloomB);
        });

    sceneImage = GPUResourceAllocator::Instance().create_image(_drawImage.imageExtent, _drawImage.imageFormat,
                                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    _mainDeletionQueue.push_function([this]() { GPUResourceAllocator::Instance().destroy_image(sceneImage); });

    ssrFeature = std::make_shared<rgraph::SSRFeature>(_device, _mainDeletionQueue, sceneData);

    ldrImage = GPUResourceAllocator::Instance().create_image(_drawImage.imageExtent, VK_FORMAT_R8G8B8A8_UNORM,
                                                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    _mainDeletionQueue.push_function([this]() { GPUResourceAllocator::Instance().destroy_image(ldrImage); });

    postFeature = std::make_shared<rgraph::PostProcessFeature>(_device, _mainDeletionQueue, "sceneImage", sceneImage, postImage, ldrImage, bloomA,
                                                               bloomB);

    rgraphInstance.AddTrackedImage("drawImage", VK_IMAGE_LAYOUT_UNDEFINED, _drawImage);
    rgraphInstance.AddTrackedImage("depthImage", VK_IMAGE_LAYOUT_UNDEFINED, _depthImage);
    rgraphInstance.AddTrackedImage("postImage", VK_IMAGE_LAYOUT_UNDEFINED, postImage);
    rgraphInstance.AddTrackedImage("sceneImage", VK_IMAGE_LAYOUT_UNDEFINED, sceneImage);
    rgraphInstance.AddTrackedImage("ldrImage", VK_IMAGE_LAYOUT_UNDEFINED, ldrImage);
    rgraphInstance.AddTrackedImage("bloomA", VK_IMAGE_LAYOUT_UNDEFINED, bloomA);
    rgraphInstance.AddTrackedImage("bloomB", VK_IMAGE_LAYOUT_UNDEFINED, bloomB);

    // the composite pass clears drawImage, so the background compute pass was pure waste
    // rgraphInstance.AddFeature(computeFeature);
    // builder.AddFeature(PBRFeature);
    // registration order is execution order: cascades must be rendered before the composite reads them.
    rgraphInstance.AddFeature(shadowFeature);
    rgraphInstance.AddFeature(localShadowFeature);
    // lightCullFeature is not added here: its pass is declared by deferredFeature, between the
    // geometry pass that fills the G-buffer and the composite that reads the grid.
    rgraphInstance.AddFeature(deferredFeature);

    // Between the deferred passes and SSR: it draws into drawImage, which SSR then reads, so the
    // overlay survives into the final image.
    lightDebugFeature = std::make_shared<rgraph::LightDebugFeature>(_device, _mainDeletionQueue, mainDrawContext, sceneData,
                                                                    _drawImage.imageFormat, _depthImage.imageFormat);
    rgraphInstance.AddFeature(lightDebugFeature);

    // after the deferred passes: reflections need the lit image to reflect
    rgraphInstance.AddFeature(ssrFeature);

    // last: everything above writes linear HDR, this resolves it to displayable LDR
    rgraphInstance.AddFeature(postFeature);

    // Street-level view of the corner cafe, captured with the P key.
    mainCamera.position = glm::vec3(-27.042f, 2.945f, 8.513f);
    mainCamera.pitch = 0.0400f;
    mainCamera.yaw = -4.4401f;

    rgraphInstance.SetTimestampPeriod(timestampPeriod);
}

void RGEngine::init_pipelines()
{
    VulkanEngine::init_pipelines();

    // no longer keeping material system on child class.
}

void RGEngine::init_default_data()
{
    VulkanEngine::init_default_data();

    MaterialSystem::MaterialResources materialResources;
    // default the material textures
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;

    GPUResourceAllocator &_gpuResourceAllocator = GPUResourceAllocator::Instance();
    // set the uniform buffer for the material data
    AllocatedBuffer materialConstants = _gpuResourceAllocator.create_buffer(sizeof(MaterialSystem::MaterialConstants),
                                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // write the buffer
    MaterialSystem::MaterialConstants *sceneUniformData = (MaterialSystem::MaterialConstants *)materialConstants.info.pMappedData;

    // freshly mapped memory is uninitialised; extra[] is read by the shaders (alpha cutoff, emissive)
    *sceneUniformData = MaterialSystem::MaterialConstants{};
    sceneUniformData->colorFactors = glm::vec4{1, 1, 1, 1};
    sceneUniformData->metal_rough_factors = glm::vec4{1, 0.5, 0, 0};

    _mainDeletionQueue.push_function([=, this]() { GPUResourceAllocator::Instance().destroy_buffer(materialConstants); });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = materialSystemInstance.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);
}

void RGEngine::cleanupOnChildren()
{

    loadedScenes.clear();
    materialSystemInstance.clear_resources(_device);
}

void RGEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    VulkanEngine::update_scene();

    sceneData.ssaoParams.x = float(ssaoSampleCount);

    loadedScenes["scene"]->Draw(glm::mat4{1.f}, mainDrawContext);

    applySunDirection();

    auto end = std::chrono::system_clock::now();

    // convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    get_current_frame().stats.scene_update_time = elapsed.count() / 1000.f;
}

/**
 * @brief Rewrite the directional light's node basis from the UI's azimuth/elevation.
 *
 * The scene graph refills DrawContext::lights every frame, so this has to run after the Draw above
 * or it would be overwritten. Doing it here rather than at each consumer means the composite, the
 * cascades and the light cull all read the same direction with no extra plumbing -- they already
 * agree that a directional light's +Z column points towards the light.
 *
 * Note sceneData.sunlightDirection is NOT the sun: only the disabled forward path reads it.
 */
void RGEngine::applySunDirection()
{
    for (GPULightingData &light : mainDrawContext.lights)
    {
        if (light.type != 0)
        {
            continue;
        }

        // Seed from the asset once, so the widget opens exactly where the glTF put the sun and
        // touching nothing changes nothing.
        if (!mainDrawContext.sunSeeded)
        {
            mainDrawContext.sunDir = glm::normalize(glm::vec3(light.transform[2]));
            mainDrawContext.sunSeeded = true;
        }

        const float len = glm::length(mainDrawContext.sunDir);
        const glm::vec3 dir = len > 1e-6f ? mainDrawContext.sunDir / len : glm::vec3(0.f, 1.f, 0.f);

        // Keep the basis orthonormal rather than writing column 2 alone: the reference axis has to
        // dodge the degenerate case of the sun directly overhead.
        const glm::vec3 ref = std::abs(dir.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        const glm::vec3 right = glm::normalize(glm::cross(ref, dir));
        const glm::vec3 up = glm::cross(dir, right);

        light.transform = glm::mat4(glm::vec4(right, 0.f), glm::vec4(up, 0.f), glm::vec4(dir, 0.f), light.transform[3]);

        // Publish to sceneData so the sky shader can reach it. This field used to be vestigial --
        // only the disabled forward path read it -- and it is the only route the sun has into
        // ssr.comp and the transparent pass, which never see the light buffer.
        // .w is the sun's strength, which the sky needs as well as its direction: a high sun at
        // 0.02 intensity is a night scene, and a Preetham midday sky over it looks broken.
        sceneData.sunlightDirection = glm::vec4(dir, mainDrawContext.sunIntensityScale);

        // sunlightColor was unused by every live shader; it now carries the HDRI controls.
        // Intensity 0 makes the sky shader take the analytic path, so it doubles as the toggle.
        sceneData.sunlightColor.x = _skyHDRILoaded ? mainDrawContext.skyHDRIIntensity : 0.0f;
        sceneData.sunlightColor.y = mainDrawContext.skyHDRIYaw;
    }
}

void RGEngine::createMsaaImages()
{
    VkExtent3D imageExtent = {_windowExtent.width, _windowExtent.height, 1};

    msaaColor.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    msaaColor.imageExtent = imageExtent;

    VkImageUsageFlags colorImageUses{};
    colorImageUses |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(msaaColor.imageFormat, colorImageUses, imageExtent, VK_SAMPLE_COUNT_8_BIT);

    // we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    GPUResourceAllocator &_gpuResourceAllocator = GPUResourceAllocator::Instance();
    _gpuResourceAllocator.create_image(&rimg_info, &rimg_allocinfo, &msaaColor.image, &msaaColor.allocation, nullptr);

    // build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(msaaColor.imageFormat, msaaColor.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &msaaColor.imageView));

    // Now creating the MSAA depth image.

    msaaDepth.imageFormat = VK_FORMAT_D32_SFLOAT;
    msaaDepth.imageExtent = imageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(msaaDepth.imageFormat, depthImageUsages, imageExtent, VK_SAMPLE_COUNT_8_BIT);

    // allocate and create the image
    _gpuResourceAllocator.create_image(&dimg_info, &rimg_allocinfo, &msaaDepth.image, &msaaDepth.allocation, nullptr);

    // build a image-view for the depth image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(msaaDepth.imageFormat, msaaDepth.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &msaaDepth.imageView));

    _mainDeletionQueue.push_function(
        [=, this]()
        {
            auto &_gpuResourceAllocator = GPUResourceAllocator::Instance();
            vkDestroyImageView(_device, msaaColor.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(msaaColor.image, msaaColor.allocation);

            vkDestroyImageView(_device, msaaDepth.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(msaaDepth.image, msaaDepth.allocation);
        });
}

void RGEngine::draw()
{
    update_scene();

    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

    // performance stuff.
    if (get_current_frame().timestampCount > 0)
    {
        rgraphInstance.ReadTimestamps(get_current_frame());
    }

    lastCompleteStats = std::move(get_current_frame().stats);
    get_current_frame().stats = {};

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);
    uint32_t swapchainImageIndex;
    // note that the _renderSemaphore will be signaled once the image is available
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._renderSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR || e == VK_SUBOPTIMAL_KHR)
    {
        resize_requested = true;
        return;
    }

    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    rgraphInstance.Build(get_current_frame()); // could potentially move this higher up to do some stuff before waiting on the current frame's fence?
    rgraphInstance.Run(get_current_frame());

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, postImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, postImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // set swapchain image layout to Attachment Optimal so we can draw it
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // draw imgui into the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // set swapchain image layout to Present so we can draw it
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));
    // end command buffer recording -----------------------

    // start submit queue -------------------------------------
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    // wait until the image has been acquired to start drawing to it.
    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._renderSemaphore);

    // signal _presentSemaphore once the queue has completed (that means the frame is ready to be presented)
    VkSemaphoreSubmitInfo signalInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, swapchainSyncStructures[swapchainImageIndex]._presentSemaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    // end submit queue ---------------------------------------

    // start present ---------------------------------

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &swapchainSyncStructures[swapchainImageIndex]._presentSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        resize_requested = true;
    }

    // increase the number of frames drawn
    _frameNumber++;

    // end present -------------------------------------
}

void RGEngine::imGuiAddParams()
{
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const float panelWidth = 340.f;

    ImVec2 panelSize(panelWidth, viewport->WorkSize.y - 20.f);

    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - panelWidth - 10.f, viewport->WorkPos.y + 10.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.95f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus;

    // passing nullptr for p_open keeps the close button off; the title bar arrow collapses the panel
    if (ImGui::Begin("RenderGraph", nullptr, flags))
    {
        ImGui::SeparatorText("Overview");
        ImGui::Columns(2, nullptr, false);
        ImGui::Text("GPU Total");
        ImGui::NextColumn();
        ImGui::Text("%.3f ms", lastCompleteStats.totalGPUTime);
        ImGui::NextColumn();
        ImGui::Text("CPU Total");
        ImGui::NextColumn();
        ImGui::Text("%.3f ms", lastCompleteStats.CPUFrametime);
        ImGui::NextColumn();
        ImGui::Columns(1);

        ImGui::Spacing();
        ImGui::SeparatorText("Render Passes");

        for (auto &pass : lastCompleteStats.passStats)
        {
            bool isCompute = pass.computeDispatches > 0;

            ImGui::PushID(pass.name.c_str());
            if (ImGui::TreeNode(pass.name.c_str()))
            {
                ImGui::Columns(2, nullptr, false);
                ImGui::Text("GPU");
                ImGui::NextColumn();
                ImGui::Text("%.3f ms", pass.GPUTime);
                ImGui::NextColumn();
                ImGui::Text("CPU");
                ImGui::NextColumn();
                ImGui::Text("%.3f ms", pass.CPUTime);
                ImGui::NextColumn();

                if (isCompute)
                {
                    ImGui::Text("Dispatches");
                    ImGui::NextColumn();
                    ImGui::Text("%.0f", pass.computeDispatches);
                    ImGui::NextColumn();
                }
                else if (pass.draws > 0)
                {
                    ImGui::Text("Draw Calls");
                    ImGui::NextColumn();
                    ImGui::Text("%.0f", pass.draws);
                    ImGui::NextColumn();
                    ImGui::Text("Triangles");
                    ImGui::NextColumn();
                    ImGui::Text("%.0f", pass.triangles);
                    ImGui::NextColumn();
                }

                ImGui::Columns(1);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::Separator();

    // The graph rebuilds every frame, so each of these takes effect on the next frame with no
    // invalidation path or pipeline rebuild.
    if (ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen))
    {
        rgraph::ShadowSettings &s = shadowFeature->settings;
        ImGui::Checkbox("Enabled", &s.enabled);
        ImGui::SliderFloat("PCF radius", &s.pcfRadius, 0.0f, 4.0f);
        ImGui::SliderFloat("Max distance", &s.maxDistance, 20.0f, 400.0f);
        ImGui::SliderFloat("Split lambda", &s.cascadeSplitLambda, 0.0f, 1.0f);
    }

    if (ImGui::CollapsingHeader("Local shadows", ImGuiTreeNodeFlags_DefaultOpen))
    {
        rgraph::LocalShadowSettings &l = localShadowFeature->settings;
        ImGui::Checkbox("Point shadows", &l.enabled);
        ImGui::Checkbox("Cull offscreen lights", &l.cullOffscreen);
        ImGui::SliderInt("Shadow casters", &l.maxLights, 0, int(rgraph::LocalShadowFeature::MAX_LIGHTS));
        ImGui::SliderFloat("Local normal bias", &l.normalBias, 0.0f, 0.5f);
        ImGui::SliderFloat("Local PCF", &l.pcfRadius, 0.0f, 4.0f);
    }

    if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("SSAO samples", (int *)&ssaoSampleCount, 0, 32);
        ImGui::SliderFloat("SSAO radius", &sceneData.ssaoParams.y, 0.1f, 10.0f);
        ImGui::SliderFloat("SSAO strength", &sceneData.ssaoParams.z, 0.0f, 3.0f);
        ImGui::SliderFloat("Ambient intensity", &sceneData.ssaoParams.w, 0.0f, 3.0f);
        ImGui::ColorEdit3("Sky", &sceneData.ambientColor.x);
        ImGui::SliderFloat("Emissive", &sceneData.debugParams.y, 0.0f, 4.0f);
    }

    if (ImGui::CollapsingHeader("Debug view", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static const char *modes[] = {"Off",       "Albedo",   "Normal",   "SSAO",         "Shadow",
                                      "Cascade",   "Roughness", "Metallic", "Emissive",    "Shadow atlas",
                                      "Tile lights"};
        int mode = int(sceneData.debugParams.x);
        if (ImGui::Combo("Mode", &mode, modes, IM_ARRAYSIZE(modes)))
        {
            sceneData.debugParams.x = float(mode);
        }

        // debug views bypass the tonemapper (and FXAA) so their values stay readable
        postFeature->settings.passthrough = mode != 0;

        // A/B the tile grid against walking all 97 lights. The image must not change.
        bool bypassCull = sceneData.debugParams.z > 0.5f;
        if (ImGui::Checkbox("Bypass light cull", &bypassCull))
        {
            sceneData.debugParams.z = bypassCull ? 1.0f : 0.0f;
        }

        // Wireframe spheres at each light's position and range. Depth-tested, so a sphere that
        // vanishes into a wall is a light buried in geometry.
        rgraph::LightDebugSettings &ld = lightDebugFeature->settings;
        ImGui::Checkbox("Show light spheres", &ld.enabled);
        if (ld.enabled)
        {
            ImGui::Checkbox("Include directional", &ld.showDirectional);
            ImGui::SliderFloat("Sphere radius scale", &ld.radiusScale, 0.1f, 5.0f);
        }
    }

    if (ImGui::CollapsingHeader("Reflections", ImGuiTreeNodeFlags_DefaultOpen))
    {
        rgraph::SSRSettings &r = ssrFeature->settings;
        ImGui::Checkbox("SSR", &r.enabled);
        ImGui::SliderFloat("Max roughness", &r.maxRoughness, 0.0f, 1.0f);
        ImGui::SliderInt("Steps", &r.steps, 4, 64);
        ImGui::SliderFloat("Thickness", &r.thickness, 0.01f, 2.0f);
        ImGui::SliderFloat("Ray distance", &r.maxDistance, 1.0f, 100.0f);
        ImGui::SliderFloat("SSR intensity", &r.intensity, 0.0f, 2.0f);
        ImGui::Checkbox("Show reflection only", &r.showReflectionOnly);
    }

    if (ImGui::CollapsingHeader("Post", ImGuiTreeNodeFlags_DefaultOpen))
    {
        rgraph::PostSettings &p = postFeature->settings;

        ImGui::Checkbox("FXAA", &p.fxaa);
        ImGui::SliderFloat("Exposure (EV)", &p.exposureEV, -6.0f, 6.0f, "%+.2f");
        ImGui::SliderFloat("Bloom", &p.bloomIntensity, 0.0f, 2.0f);
        ImGui::SliderFloat("Bloom threshold", &p.bloomThreshold, 0.0f, 8.0f);
        ImGui::SliderFloat("Bloom clamp", &p.bloomClamp, 0.5f, 64.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    }

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // glTF light units vary wildly between exporters, so this is a manual dial rather than a
        // baked-in photometric conversion.
        ImGui::SliderFloat("Sun intensity", &mainDrawContext.sunIntensityScale, 0.0f, 20.0f);
        ImGui::SliderFloat("Lamp intensity", &mainDrawContext.localIntensityScale, 0.0f, 20.0f);

        // The gizmo is driven in VIEW space, not world space. Its sphere then stands for the
        // screen you are looking at: drag the arrow up and the sun rises in shot, drag it towards
        // you and the sun ends up behind the camera lighting the scene head on. Fed world space
        // instead, the same drag means something different every time the camera turns, which is
        // useless for actually placing a sun against a shot.
        //
        // The view matrix's upper 3x3 is orthonormal, so transpose == inverse. The Y flip lives in
        // the projection rather than the view, so view-space +Y really is up on screen.
        const glm::mat3 viewRot = glm::mat3(sceneData.view);
        glm::vec3 sunView = viewRot * mainDrawContext.sunDir;

        if (ImGui::gizmo3D("##SunDir", sunView, 110, imguiGizmo::modeDirection))
        {
            mainDrawContext.sunDir = glm::transpose(viewRot) * sunView;
        }

        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::TextDisabled("sun (world)");
        ImGui::TextDisabled("%.2f %.2f %.2f", mainDrawContext.sunDir.x, mainDrawContext.sunDir.y, mainDrawContext.sunDir.z);
        ImGui::EndGroup();

        if (_skyHDRILoaded)
        {
            ImGui::SliderFloat("HDRI intensity", &mainDrawContext.skyHDRIIntensity, 0.0f, 5.0f);
            ImGui::SliderFloat("HDRI rotation", &mainDrawContext.skyHDRIYaw, -3.1416f, 3.1416f, "%.2f rad");
            ImGui::TextDisabled("intensity 0 = analytic sky");
        }
        else
        {
            ImGui::TextDisabled("no assets/sky.hdr - using analytic sky");
        }
        ImGui::SliderFloat("Camera speed", &mainCamera.speed, 0.5f, 500.0f, "%.1f u/s", ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled("press P to dump camera pos/pitch/yaw to stdout");
        ImGui::Text("lights: %zu   opaque: %zu", mainDrawContext.lights.size(), mainDrawContext.OpaqueSurfaces.size());
    }

    ImGui::End();
}