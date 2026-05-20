#include "vk_engine.h"
#include "SDL_events.h"
#include "SDL_scancode.h"
#include "fmt/base.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

// ---- other includes ----
#include "VkBootstrap.h"
#include "vk_images.h"
#include "vk_loader.h"
#include <GPUResourceAllocator.h>
#include <glm/gtx/transform.hpp>
#include <vk_images.h>
#include <vk_initializers.h>
#include <vk_types.h>

#include <vk_mem_alloc.h>

#include <vk_pipelines.h>
// imgui
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

VulkanEngine *VulkanEngine::loadedEngine = nullptr;

// forward declaration - this is present in Scenegraphs.cpp
bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj);

VulkanEngine &VulkanEngine::Instance()
{
    return *loadedEngine;
}
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window =
        SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width, _windowExtent.height, window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_default_data();

    init_imgui();

    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(30.f, -00.f, -085.f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 2.5f;

    // everything went fine
    _isInitialized = true;
}

// {{{ INITIALIZATION

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    auto instRet =
        builder.set_app_name("Vulkan Engine").request_validation_layers(true).use_default_debug_messenger().require_api_version(1, 3, 0).build();

    vkb::Instance vkbInst = instRet.value();
    _instance = vkbInst.instance;
    _debugMessenger = vkbInst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .synchronization2 = true, .dynamicRendering = true};

    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .descriptorIndexing = true, .bufferDeviceAddress = true};

    vkb::PhysicalDeviceSelector selector{vkbInst};

    vkb::PhysicalDevice PhysicalDevice = selector.set_minimum_version(1, 3)
                                             .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                                             .allow_any_gpu_device_type(false)
                                             .set_required_features_13(features)
                                             .set_required_features_12(features12)
                                             .set_surface(_surface)
                                             .add_required_extension("VK_KHR_shader_relaxed_extended_instruction")
                                             .select()
                                             .value();

    vkb::DeviceBuilder DeviceBuilder{PhysicalDevice};

    VkPhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR relaxedInstFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES_KHR, .shaderRelaxedExtendedInstruction = true};

    vkb::Device vkbDevice = DeviceBuilder.add_pNext(&relaxedInstFeatures).build().value();

    _device = vkbDevice.device;
    _chosenGPU = PhysicalDevice.physical_device;
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    GPUResourceAllocator::init(_allocator, _device);

    _mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    // draw image size will match the window
    VkExtent3D drawImageExtent = {_windowExtent.width, _windowExtent.height, 1};

    // hardcoding the draw format to 32 bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    GPUResourceAllocator _gpuResourceAllocator = GPUResourceAllocator::Instance();
    _gpuResourceAllocator.create_image(&rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    // build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    // similarly, create the depth image

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // allocate and create the image
    _gpuResourceAllocator.create_image(&dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    // build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    // add to deletion queues
    _mainDeletionQueue.push_function(
        [=, this]()
        {
            auto _gpuResourceAllocator = GPUResourceAllocator::Instance();
            vkDestroyImageView(_device, _drawImage.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(

                _drawImage.image, _drawImage.allocation);

            vkDestroyImageView(_device, _depthImage.imageView, nullptr);
            _gpuResourceAllocator.destroy_image(_depthImage.image, _depthImage.allocation);
        });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder SwapchainBuilder{_chosenGPU, _device, _surface};

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain =
        SwapchainBuilder.set_desired_format(VkSurfaceFormatKHR{.format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::init_commands()
{

    // create command pool
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

        // allocate stuff for performance monitoring.
        VkQueryPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = _frames[i].maxTimestamps;

        VK_CHECK(vkCreateQueryPool(_device, &poolInfo, nullptr, &_frames[i].timestampQueryPool));
    }

    // imgui
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(_chosenGPU, &props);
    timestampPeriod = props.limits.timestampPeriod;

    _mainDeletionQueue.push_function([=, this]() { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }

    swapchainSyncStructures.resize(_swapchainImages.size());

    for (int i = 0; i < swapchainSyncStructures.size(); i++)
    {
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &swapchainSyncStructures[i]._presentSemaphore));
    }
    // imgui
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=, this]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::init_descriptors()
{
    // create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};

    globalDescriptorAllocator.init(_device, 10, sizes);

    // make the descriptor set layout for our compute draw
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // allocate a descriptor set for our draw image
    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    writer.update_set(_device, _drawImageDescriptors);

    // global descriptor sets
    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() { _frames[i]._frameDescriptors.destroy_pools(_device); });
    }

    // make sure both the descriptor allocator and the new layout get cleaned up properly
    _mainDeletionQueue.push_function(
        [&]()
        {
            globalDescriptorAllocator.destroy_pools(_device);

            vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        });
}

void VulkanEngine::init_pipelines()
{
    // compute
    init_background_pipelines();

    materialSystemInstance.build_descriptors(_device);
    // mesh piplines are no longer used, and the gltf pipelines are built with the rendergraph now.
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../shaders/compute/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../shaders/compute/sky.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = gradientShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    // change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    // default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    // add the 2 background effects into the array
    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    // destroy structures properly
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function(
        [=, this]()
        {
            vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
            vkDestroyPipeline(_device, sky.pipeline, nullptr);
            vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        });
}

void VulkanEngine::init_default_data()
{

    GPUResourceAllocator _gpuResourceAllocator = GPUResourceAllocator::Instance();
    // 3 default textures, white, grey, black. 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = _gpuResourceAllocator.create_image((void *)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = _gpuResourceAllocator.create_image((void *)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = _gpuResourceAllocator.create_image((void *)&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels; // for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++)
    {
        for (int y = 0; y < 16; y++)
        {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage =
        _gpuResourceAllocator.create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    _mainDeletionQueue.push_function(
        [=, this]()
        {
            vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
            vkDestroySampler(_device, _defaultSamplerLinear, nullptr);
            GPUResourceAllocator _gpuResourceAllocator = GPUResourceAllocator::Instance();
            _gpuResourceAllocator.destroy_image(_whiteImage);
            _gpuResourceAllocator.destroy_image(_greyImage);
            _gpuResourceAllocator.destroy_image(_blackImage);
            _gpuResourceAllocator.destroy_image(_errorCheckerboardImage);
        });

    // gltf resources are now created in PBREngine.cpp
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    // dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function(
        [=, this]()
        {
            ImGui_ImplVulkan_Shutdown();
            vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        });
}

// }}} END INITIALIZATION

// {{{ DESTRUCTION AND CLEANUP

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {

        vkDeviceWaitIdle(_device);

        // loadedScenes.clear();

        for (int i = 0; i < FRAME_OVERLAP; i++)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

            // Destroy query pools
            if (_frames[i].timestampQueryPool != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(_device, _frames[i].timestampQueryPool, nullptr);
            }

            // destroy sync objects
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }
        for (int i = 0; i < swapchainSyncStructures.size(); i++)
        {
            vkDestroySemaphore(_device, swapchainSyncStructures[i]._presentSemaphore, nullptr);
        }

        cleanupOnChildren();

        _mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::cleanupOnChildren()
{
    // intentionally left blank to cleanup on children
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (int i = 0; i < _swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

// }}} END DESTRUCTION AND CLEANUP

// {{{ MISC IMPORTANT

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

// }}} END MISC IMPORTANT

// {{{ DRAW

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    SDL_Scancode scanCode;

    // main loop
    while (!bQuit)
    {
        auto start = std::chrono::system_clock::now();

        // Handle events on queue
        while (SDL_PollEvent(&e) != 0)
        {

            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
            {
                bQuit = true;
            }

            if (e.type == SDL_WINDOWEVENT)
            {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED)
                {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED)
                {
                    stop_rendering = false;
                }
            }

            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == 'p')
            {
                auto val = mainCamera.getCameraPos();
                fmt::println("camera pos : {} {} {}", val.x, val.y, val.z);
            }

            mainCamera.processSDLEvent(e);

            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        if (resize_requested)
        {
            resize_swapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // if (ImGui::Begin("Stats"))
        // {
        //     ImGui::Text("Framerate: %.4f", 1 / lastCompleteStats.frameTime * 1000);
        //     ImGui::Text("frametime %.4f ms", lastCompleteStats.frameTime);
        //     ImGui::Text("update time %.4f ms", lastCompleteStats.scene_update_time);
        // }
        // ImGui::End();

        imGuiAddParams();

        // make imgui calculate internal draw structures
        ImGui::Render();

        // do not draw if we are minimized
        if (stop_rendering)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();

        auto end = std::chrono::system_clock::now();

        // convert to microseconds (integer), and then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        get_current_frame().stats.frameTime = elapsed.count() / 1000.f;
    }
}

void VulkanEngine::draw()
{
    update_scene();

    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
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

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    // start command buffer recording ---------------------
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // setup depth attachment
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // draw_geometry(cmd);

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

void VulkanEngine::update_scene()
{
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();
    mainDrawContext.lights.clear();

    mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();

    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;

    // some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);
    sceneData.cameraPos = mainCamera.getCameraPos();
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{

    ComputeEffect &effect = backgroundEffects[currentBackgroundEffect];

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

// }}} END DRAW
