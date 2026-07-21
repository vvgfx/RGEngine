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
#include <algorithm>
#include <chrono>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <memory>
#include <vulkan/vulkan_core.h>

void RGEngine::init()
{

    VulkanEngine::init();

    std::string structurePath = {"../assets/outpostWithLights4.glb"};

    GLTFCreatorData creatorData = {};

    creatorData._defaultSamplerLinear = _defaultSamplerLinear;
    creatorData.defaultImage = _whiteImage;
    creatorData.loadErrorImage = _errorCheckerboardImage;
    creatorData._device = _device;
    creatorData.materialSystemReference = &materialSystemInstance;

    // this is called after the pipelines are initialzed.
    auto structureFile = loadGltf(creatorData, structurePath);

    assert(structureFile.has_value());

    loadedScenes["outpost"] = *structureFile;

    structureFile.value()->name = "outpost";

    // --- box3d physics demo setup ---
    // Two cube meshes with different colors: orange falling body, teal ground slab.
    auto cubeFile = loadGltf(creatorData, "../physics_models/box3d_cube.gltf");
    assert(cubeFile.has_value());
    loadedScenes["cube"] = *cubeFile;
    cubeFile.value()->name = "cube";

    auto groundFile = loadGltf(creatorData, "../physics_models/box3d_ground.gltf");
    assert(groundFile.has_value());
    loadedScenes["ground"] = *groundFile;
    groundFile.value()->name = "ground";

    // Physics world (default gravity {0,-10,0}).
    b3WorldDef worldDef = b3DefaultWorldDef();
    physicsWorld = b3CreateWorld(&worldDef);

    // Static ground slab: center y=-1, half-extents (20,1,20) -> top surface at y=0.
    b3BodyDef groundDef = b3DefaultBodyDef();
    groundDef.position = b3Pos{0.0, -1.0, 0.0};
    b3BodyId ground = b3CreateBody(physicsWorld, &groundDef);
    b3BoxHull groundHull = b3MakeBoxHull(20.0f, 1.0f, 20.0f);
    b3ShapeDef groundShape = b3DefaultShapeDef();
    b3CreateHullShape(ground, &groundShape, &groundHull.base);

    // Dynamic unit cube (half-extent 1, matches b3MakeCubeHull(1)) dropped from y=8.
    b3BodyDef boxDef = b3DefaultBodyDef();
    boxDef.type = b3_dynamicBody;
    boxDef.position = b3Pos{0.0, 8.0, 0.0};
    fallingBox = b3CreateBody(physicsWorld, &boxDef);
    b3BoxHull cubeHull = b3MakeCubeHull(1.0f);
    b3ShapeDef cubeShape = b3DefaultShapeDef();
    cubeShape.density = 1.0f; // dynamic bodies need non-zero density
    b3CreateHullShape(fallingBox, &cubeShape, &cubeHull.base);

    // Frame the physics scene.
    mainCamera.position = glm::vec3{0.f, 4.f, 25.f};
    mainCamera.pitch = -0.15f;
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
    // create MSAA images. TODO: move these out somewhere later.
    createMsaaImages();

    builder.AddTrackedImage("drawImage", VK_IMAGE_LAYOUT_UNDEFINED, _drawImage);
    builder.AddTrackedImage("depthImage", VK_IMAGE_LAYOUT_UNDEFINED, _depthImage);
    builder.AddTrackedImage("msaaColor", VK_IMAGE_LAYOUT_UNDEFINED, msaaColor);
    builder.AddTrackedImage("msaaDepth", VK_IMAGE_LAYOUT_UNDEFINED, msaaDepth);

    builder.AddFeature(computeFeature);
    // builder.AddFeature(PBRFeature);
    builder.AddFeature(deferredFeature);

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

    b3DestroyWorld(physicsWorld);
    loadedScenes.clear();
    materialSystemInstance.clear_resources(_device);
}

void RGEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    VulkanEngine::update_scene();

    // --- box3d: advance the simulation with a fixed-timestep accumulator ---
    auto now = std::chrono::steady_clock::now();
    float frameDt = std::chrono::duration<float>(now - lastPhysicsTime).count();
    lastPhysicsTime = now;
    if (!physicsPaused)
    {
        physicsAccumulator = std::min(physicsAccumulator + frameDt, 0.25f); // clamp: anti spiral-of-death
        const float fixedStep = 1.0f / 60.0f;
        while (physicsAccumulator >= fixedStep)
        {
            b3World_Step(physicsWorld, fixedStep, 4);
            physicsAccumulator -= fixedStep;
        }
    }

    // Read the cube pose (position is b3Pos/double; rotation is b3Quat{ b3Vec3 v; float s; }).
    b3Pos p = b3Body_GetPosition(fallingBox);
    b3Quat r = b3Body_GetRotation(fallingBox);
    glm::vec3 cubePos((float)p.x, (float)p.y, (float)p.z);
    glm::quat cubeRot(r.s, r.v.x, r.v.y, r.v.z); // glm order is (w, x, y, z)

    // Place each mesh via topMatrix (Scene::Draw bakes topMatrix * worldTransform).
    glm::mat4 cubeM = glm::translate(glm::mat4(1.f), cubePos) * glm::toMat4(cubeRot);
    glm::mat4 groundM =
        glm::translate(glm::mat4(1.f), glm::vec3(0.f, -1.f, 0.f)) * glm::scale(glm::mat4(1.f), glm::vec3(20.f, 1.f, 20.f));
    loadedScenes["cube"]->Draw(cubeM, mainDrawContext);
    loadedScenes["ground"]->Draw(groundM, mainDrawContext);

    // Standalone scene has no glTF lights, so supply one in code.
    // (intensity must be large: attenuation is 1/dist^2; range must exceed distance or the shader culls it.)
    GPULightingData light{};
    light.transform = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 15.f, 10.f));
    light.color = glm::vec3(1.f, 1.f, 1.f);
    light.intensity = 300.f;
    light.range = 1000.f;
    mainDrawContext.lights.push_back(light);

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
    if (ImGui::Begin("RenderGraph details"))
    {
        // Summary section
        ImGui::SeparatorText("RenderGraph Overview");
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
            if (ImGui::CollapsingHeader(pass.name.c_str()))
            {
                ImGui::Indent();
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
                ImGui::Unindent();
            }
            ImGui::PopID();
        }
    }
    ImGui::End();

    if (ImGui::Begin("box3d Physics"))
    {
        b3Pos p = b3Body_GetPosition(fallingBox);
        ImGui::Text("cube position: (%.2f, %.2f, %.2f)", (float)p.x, (float)p.y, (float)p.z);
        ImGui::Checkbox("Pause", &physicsPaused);
        if (ImGui::Button("Re-drop"))
        {
            b3Body_SetTransform(fallingBox, b3Pos{0.0, 8.0, 0.0}, b3Quat_identity);
            b3Body_SetLinearVelocity(fallingBox, b3Vec3{0.f, 0.f, 0.f});
            b3Body_SetAngularVelocity(fallingBox, b3Vec3{0.f, 0.f, 0.f});
            b3Body_SetAwake(fallingBox, true);
        }
    }
    ImGui::End();
}