#include "GPUResourceAllocator.h"
#include "MaterialSystem.h"
#include "imgui.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DebugDrawFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include "sgraph/ScenegraphImporter.h"
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_loader.h"
#include "vk_types.h"
#include <RGEngine.h>
#include <SDL_keyboard.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <memory>
#include <vulkan/vulkan_core.h>

void RGEngine::init()
{

    VulkanEngine::init();

    // authored scenegraph is authoritative; each glTF file enters it as a Scene node
    sgraph::ScenegraphImporter importer;
    std::ifstream sceneStream("../scenegraphs/magnet-test-1.txt");
    if (!sceneStream)
    {
        fmt::print("RGEngine: failed to open ../scenegraphs/magnet-test-1.txt\n");
    }
    scenegraph = importer.parse(sceneStream);

    // physics reads world poses, so bake first
    if (auto rootNode = std::dynamic_pointer_cast<sgraph::Node>(scenegraph->getRoot()))
    {
        rootNode->refreshTransform(glm::mat4{1.f});
    }

    physics.init();
    physics.buildFromScene(scenegraph, importer.getPhysicsSpecs(), importer.getMagnetSpecs());

    mainCamera.position = glm::vec3{0.f, 8.f, 32.f};
    mainCamera.pitch = -0.25f;
    mainCamera.yaw = 0.f;

    lastPhysicsTime = std::chrono::steady_clock::now();

    builder.Init(_device, _drawImage.imageExtent, _instance);

    VkExtent3D extent = {_windowExtent.width, _windowExtent.height, 1};
    computeFeature = std::make_shared<rgraph::ComputeBackgroundFeature>(_device, _mainDeletionQueue, extent, _drawImage);
    MaterialSystemCreateInfo msCreateInfo = {_device, _drawImage.imageFormat, _depthImage.imageFormat, _gpuSceneDataDescriptorLayout};
    PBRFeature = std::make_shared<rgraph::PBRShadingFeature>(mainDrawContext, _device, msCreateInfo, sceneData, _gpuSceneDataDescriptorLayout,
                                                             _mainDeletionQueue);

    deferredFeature = std::make_shared<rgraph::DeferredRenderingFeature>(mainDrawContext, _device, sceneData, _gpuSceneDataDescriptorLayout,
                                                                         msCreateInfo, _mainDeletionQueue);

    // fills background pixels, so it runs after the composite
    skyboxFeature = std::make_shared<rgraph::SkyboxFeature>(_device, _drawImage.imageFormat, _depthImage.imageFormat, _mainDeletionQueue);
    if (auto sky = loadImage("../physics_models/sky_18_2k.png"))
        skyboxFeature->setSkyTexture(**sky);   // optional<shared_ptr<AllocatedImage>>
    else
        skyboxFeature->setSkyTexture(_whiteImage);

    // draws over drawImage, so it must come after the deferred passes
    debugFeature = std::make_shared<rgraph::DebugDrawFeature>(_device, sceneData, _gpuSceneDataDescriptorLayout, _drawImage.imageFormat,
                                                              _depthImage.imageFormat, _mainDeletionQueue);

    // create MSAA images. TODO: move these out somewhere later.
    createMsaaImages();

    builder.AddTrackedImage("drawImage", VK_IMAGE_LAYOUT_UNDEFINED, _drawImage);
    builder.AddTrackedImage("depthImage", VK_IMAGE_LAYOUT_UNDEFINED, _depthImage);
    builder.AddTrackedImage("msaaColor", VK_IMAGE_LAYOUT_UNDEFINED, msaaColor);
    builder.AddTrackedImage("msaaDepth", VK_IMAGE_LAYOUT_UNDEFINED, msaaDepth);

    // composite clears drawImage, so the pre-pass sky was pointless
    // builder.AddFeature(computeFeature);
    // builder.AddFeature(PBRFeature);
    builder.AddFeature(deferredFeature);
    builder.AddFeature(skyboxFeature); // sky fills background pixels after lighting
    builder.AddFeature(debugFeature);  // overlay must be last so it draws on top

    builder.SetTimestampPeriod(timestampPeriod);
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

    GPUResourceAllocator _gpuResourceAllocator = GPUResourceAllocator::Instance();
    // set the uniform buffer for the material data
    AllocatedBuffer materialConstants = _gpuResourceAllocator.create_buffer(sizeof(MaterialSystem::MaterialConstants),
                                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // write the buffer
    MaterialSystem::MaterialConstants *sceneUniformData = (MaterialSystem::MaterialConstants *)materialConstants.info.pMappedData;
    sceneUniformData->colorFactors = glm::vec4{1, 1, 1, 1};
    sceneUniformData->metal_rough_factors = glm::vec4{1, 0.5, 0, 0};

    _mainDeletionQueue.push_function([=, this]() { GPUResourceAllocator::Instance().destroy_buffer(materialConstants); });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = materialSystemInstance.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);
}

void RGEngine::cleanupOnChildren()
{
    physics.cleanup(); // destroy world + heap hulls first
    if (scenegraph)
        scenegraph->cleanup();
    scenegraph.reset(); // releases glTF geometry (Scene dtors free GPU resources)
    loadedScenes.clear();
    materialSystemInstance.clear_resources(_device);
}

void RGEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    VulkanEngine::update_scene();

    // --- directional sun + sky params ---
    glm::vec3 sunDir = glm::normalize(sunDirection);
    sceneData.sunlightDirection = glm::vec4(sunDir, sunIntensity);
    sceneData.sunlightColor = glm::vec4(sunColor, 1.f);
    sceneData.ambientColor = glm::vec4(skyZenith * 0.15f, 1.f); // ambient tracks the sky

    // an orthographic "camera" at the sun
    {
        glm::vec3 up = std::abs(sunDir.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::mat4 lightView = glm::lookAt(-sunDir * 40.f, glm::vec3(0.f), up);
        glm::mat4 lightProj = glm::ortho(-25.f, 25.f, -25.f, 25.f, 0.1f, 100.f);
        lightProj[1][1] *= -1.f; // match the engine's Vulkan-Y convention
        sceneData.sunViewProj = lightProj * lightView;
        sceneData.shadowParams = glm::vec4(shadowBias, shadowsEnabled ? 1.f : 0.f, 0.f, 0.f);
    }
    if (skyboxFeature)
    {
        rgraph::SkyboxFeature::Params p;
        p.invViewProj = glm::inverse(sceneData.viewproj);
        p.cameraPos = sceneData.cameraPos;
        p.sunDirection = glm::vec4(sunDir, sunIntensity);
        p.sunColor = glm::vec4(sunColor, sunDiskSize);
        p.horizon = glm::vec4(skyHorizon, 1.f);
        p.zenith = glm::vec4(skyZenith, 1.f);
        p.ground = glm::vec4(skyGround, 1.f);
        p.mode = glm::ivec4(skyMode, 0, 0, 0);
        skyboxFeature->setParams(p);
    }

    // held, not KEYDOWN: one event would only feed a single substep
    const Uint8 *keys = SDL_GetKeyboardState(nullptr);
    physics.debugForceActive = keys[SDL_SCANCODE_SPACE] != 0;

    // fixed timestep, so the sim is independent of frame rate
    auto now = std::chrono::steady_clock::now();
    float frameDt = std::chrono::duration<float>(now - lastPhysicsTime).count();
    lastPhysicsTime = now;
    if (!physicsPaused)
    {
        physics.step(frameDt * physicsTimeScale);
    }
    physics.sync();

    if (scenegraph && scenegraph->getRoot())
    {
        scenegraph->getRoot()->Draw(glm::mat4{1.f}, mainDrawContext);
    }

    // no glTF lights in the authored scene. range must exceed distance or the shader culls it
    GPULightingData light{};
    light.transform = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 20.f, 15.f));
    light.color = glm::vec3(1.f, 1.f, 1.f);
    light.intensity = 400.f;
    light.range = 2000.f;
    mainDrawContext.lights.push_back(light);

    if (debugFeature)
    {
        debugFeature->enabled = showDebugDraw || showMagnets || showMagnetForces;
        std::vector<DebugLineVertex> lines;
        if (showDebugDraw)
        {
            physics.drawDebug(lines);
        }
        if (showMagnets)
        {
            physics.drawMagnets(lines);
        }
        if (showMagnetForces)
        {
            physics.drawMagnetForces(lines);
        }
        debugFeature->setLines(std::move(lines));
    }

    auto end = std::chrono::system_clock::now();

    // convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    get_current_frame().stats.scene_update_time = elapsed.count() / 1000.f;
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
        builder.ReadTimestamps(get_current_frame());
    }

    lastCompleteStats = get_current_frame().stats;
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

    builder.Build(get_current_frame()); // could potentially move this higher up to do some stuff before waiting on the current frame's fence?
    builder.Run(get_current_frame());

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

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
    const ImGuiIO &io = ImGui::GetIO();
    const float panelWidth = 340.f;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - panelWidth - 10.f, 10.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, io.DisplaySize.y - 20.f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Controls"))
    {
        if (ImGui::CollapsingHeader("RenderGraph", ImGuiTreeNodeFlags_DefaultOpen))
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

        if (ImGui::CollapsingHeader("Sky / Sun"))
        {
            const char *modes[] = {"Procedural", "Texture"};
            ImGui::Combo("Sky mode", &skyMode, modes, 2);
            ImGui::SeparatorText("Sun");
            ImGui::DragFloat3("Direction", &sunDirection.x, 0.01f, -1.f, 1.f);
            ImGui::ColorEdit3("Color", &sunColor.x);
            ImGui::DragFloat("Intensity", &sunIntensity, 0.1f, 0.f, 50.f);
            ImGui::DragFloat("Disk size", &sunDiskSize, 0.001f, 0.f, 0.5f);
            ImGui::SeparatorText("Sky gradient");
            ImGui::ColorEdit3("Horizon", &skyHorizon.x);
            ImGui::ColorEdit3("Zenith", &skyZenith.x);
            ImGui::ColorEdit3("Ground", &skyGround.x);
        }

        if (ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Enable shadows", &shadowsEnabled);
            ImGui::DragFloat("Shadow bias", &shadowBias, 0.0005f, 0.f, 0.05f, "%.4f");
        }

        if (ImGui::CollapsingHeader("Physics"))
        {
            ImGui::Text("bodies: %zu", physics.bodyCount());
            ImGui::Checkbox("Show collider wireframes", &showDebugDraw);
            ImGui::Checkbox("Pause", &physicsPaused);
            ImGui::SameLine();
            if (ImGui::Button("Step once"))
            {
                physics.stepOnce();
            }
            ImGui::DragFloat("Time scale", &physicsTimeScale, 0.005f, 0.01f, 1.f, "%.3f");
            if (ImGui::Button("Re-drop"))
            {
                physics.reset();
            }

            ImGui::SeparatorText("Debug force");
            std::vector<std::string> names = physics.bodyNames();
            if (physics.debugTarget.empty() && !names.empty())
            {
                auto head = std::find(names.begin(), names.end(), "m_head");
                physics.debugTarget = (head != names.end()) ? *head : names.front();
            }
            if (ImGui::BeginCombo("Target", physics.debugTarget.c_str()))
            {
                for (const std::string &name : names)
                {
                    if (ImGui::Selectable(name.c_str(), name == physics.debugTarget))
                    {
                        physics.debugTarget = name;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Text("mass: %.2f", physics.bodyMass(physics.debugTarget));
            ImGui::DragFloat3("Force (world, xN weight)", &physics.debugForceWeights.x, 0.1f, -20.f, 20.f);
            ImGui::Text("hold SPACE to apply%s", physics.debugForceActive ? "   [ACTIVE]" : "");

            ImGui::SeparatorText("Magnets");
            ImGui::Checkbox("Show magnets", &showMagnets);
            ImGui::Checkbox("Enable magnets", &physics.magnetsEnabled);
            ImGui::DragFloat("Pole strength", &physics.magnetStrength, 0.01f, 0.f, 50.f, "%.3f");
            ImGui::Checkbox("Show force vectors", &showMagnetForces);
            ImGui::DragFloat("Arrow scale", &physics.magnetArrowScale, 1.0e-4f, 0.f, 1.f, "%.5f");
            ImGui::Text("pairs last step: %d", physics.magnetPairsLastStep);
            ImGui::Text("largest pole force: %.1f", physics.largestPoleForce());
        }
    }
    ImGui::End();
}