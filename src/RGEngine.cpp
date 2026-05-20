#include "GPUResourceAllocator.h"
#include "MaterialSystem.h"
#include "imgui.h"
#include "rgraph/features/ComputeBackgroundFeature.h"
#include "rgraph/features/DeferredRenderingFeature.h"
#include "rgraph/features/PBRShadingFeature.h"
#include "rgraph/features/SMAAFeature.h"
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_loader.h"
#include "vk_types.h"
#include <RGEngine.h>
#include <chrono>
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
    createSmaaOutput();

    VkExtent3D extent3D = {_windowExtent.width, _windowExtent.height, 1};
    smaaFeature = std::make_shared<rgraph::SMAAFeature>(_device, extent3D, _mainDeletionQueue);

    builder.AddTrackedImage("drawImage", VK_IMAGE_LAYOUT_UNDEFINED, _drawImage);
    builder.AddTrackedImage("depthImage", VK_IMAGE_LAYOUT_UNDEFINED, _depthImage);
    builder.AddTrackedImage("msaaColor", VK_IMAGE_LAYOUT_UNDEFINED, msaaColor);
    builder.AddTrackedImage("msaaDepth", VK_IMAGE_LAYOUT_UNDEFINED, msaaDepth);
    builder.AddTrackedImage("smaaOutput", VK_IMAGE_LAYOUT_UNDEFINED, smaaOutput);

    builder.AddFeature(computeFeature);
    // builder.AddFeature(PBRFeature);
    builder.AddFeature(deferredFeature);
    builder.AddFeature(smaaFeature);

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

    loadedScenes.clear();
    materialSystemInstance.clear_resources(_device);
}

void RGEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    VulkanEngine::update_scene();

    loadedScenes["outpost"]->Draw(glm::mat4{1.f}, mainDrawContext);

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

void RGEngine::createSmaaOutput()
{
    VkExtent3D imageExtent = {_windowExtent.width, _windowExtent.height, 1};

    VkImageUsageFlags usages =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    GPUResourceAllocator &gpuAlloc = GPUResourceAllocator::Instance();
    smaaOutput = gpuAlloc.create_image(imageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, usages);

    _mainDeletionQueue.push_function(
        [this]()
        {
            GPUResourceAllocator::Instance().destroy_image(smaaOutput);
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
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._renderSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR || e == VK_SUBOPTIMAL_KHR)
    {
        resize_requested = true;
        return;
    }

    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    builder.Build(get_current_frame());
    builder.Run(get_current_frame());

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // SMAA neighborhood blend leaves smaaOutput in COLOR_ATTACHMENT_OPTIMAL; transition to blit source.
    vkutil::transition_image(cmd, smaaOutput.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Blit the SMAA-resolved image to the swapchain.
    vkutil::copy_image_to_image(cmd, smaaOutput.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

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

    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._renderSemaphore);
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
}