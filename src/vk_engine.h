// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "MaterialSystem.h"
#include <GPUResourceAllocator.h>
#include <camera.h>
#include <cstdint>
#include <functional>
#include <vector>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <vk_types.h>

struct PassStats
{
    std::string name;
    float GPUTime;
    float CPUTime;
    // compute details.
    float computeDispatches = 0;
    float triangles = 0;
    float draws = 0;
};

struct EngineStats
{
    float frameTime;
    float CPUFrametime;
    float totalGPUTime;
    float scene_update_time;
    std::vector<PassStats> passStats;
};

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()> &&function)
    {
        deletors.push_back(function);
    }

    void flush()
    {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
        {
            (*it)(); // call functors
        }

        deletors.clear();
    }
};

struct FrameData
{
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
    VkSemaphore _renderSemaphore; //, _renderSemaphore;
    VkFence _renderFence;
    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;

    // performance stuff.
    // GPU stuff first
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    uint32_t maxTimestamps = 64; // 32 passes * 2 timestamps each
    uint32_t timestampCount = 0;
    std::vector<std::pair<std::string, uint32_t>> passIndices;
    std::pair<uint32_t, uint32_t> totalTimeIndices;

    // store performance data
    EngineStats stats;
};

struct SyncStructures
{
    VkSemaphore _presentSemaphore;
};

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect
{
    const char *name;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    ComputePushConstants data;
};

struct GPUSceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
    glm::vec4 cameraPos;
    glm::mat4 sunViewProj;   // world -> sun light clip space (directional shadow map)
    glm::vec4 shadowParams;  // x = depth bias, y = shadows enabled (0/1)
};

// GPU lighting data required for punctual lights from GLTF.
struct GPULightingData
{
    glm::mat4 transform; // spotlights are pointed towards the -ve z direction, so use the transform to find the
                         // direction in world space.
    glm::vec3 color;
    float intensity;
    float range;
};

// {{{ SCENEGRAPHS --------------------------

struct RenderObject
{
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance *material;
    Bounds bounds;
    glm::mat4 modelMatrix;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
    std::vector<GPULightingData> lights;
};

// }}} SCENEGRAPHS end -----------------------

constexpr unsigned int FRAME_OVERLAP = 3;

class VulkanEngine
{
  public:
    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{2560, 1440};
    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debugMessenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    FrameData _frames[FRAME_OVERLAP];
    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;
    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator;
    AllocatedImage _drawImage;
    AllocatedImage _depthImage; // depth testing
    VkExtent2D _drawExtent;
    DescriptorAllocatorGrowable globalDescriptorAllocator;
    MaterialSystem materialSystemInstance;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    // pipeline and shader modules
    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    // imgui stuff
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;
    void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

    // multiple compute pipelines
    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{1};

    // mesh pipelines
    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;

    bool resize_requested;
    float renderScale = 1.f;

    // scene data
    GPUSceneData sceneData;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    // default images
    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

    // default materials
    MaterialInstance defaultData;

    // sync structure - removed from frameData
    std::vector<SyncStructures> swapchainSyncStructures;

    // timestamps for frame data.
    float timestampPeriod;

    FrameData &get_current_frame()
    {
        return _frames[_frameNumber % FRAME_OVERLAP];
    };

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    struct SDL_Window *_window{nullptr};

    static VulkanEngine &Instance();

    // initializes everything in the engine
    virtual void init();

    // shuts down the engine
    void cleanup();

    // draw loop
    virtual void draw();

    // run main loop
    void run();

    virtual void imGuiAddParams()
    {
    }

    MaterialSystem &GetMaterialSystem()
    {
        return materialSystemInstance;
    }

    VkExtent2D GetWindowExtent()
    {
        return _windowExtent;
    }

    VkDevice GetVkDevice()
    {
        return _device;
    }

    // scenegraph stuff

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<sgraph::Node>> loadedNodes;

    virtual void update_scene();

    // camera
    Camera mainCamera;

    // statistics
    EngineStats lastCompleteStats;

  protected:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void draw_background(VkCommandBuffer cmd);

    void init_descriptors();

    // pipeline stuff
    virtual void init_pipelines();
    void init_background_pipelines();

    // imgui
    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    virtual void init_default_data();

    void resize_swapchain();

    // single cleanup override method for all children
    virtual void cleanupOnChildren();

  private:
    static VulkanEngine *loadedEngine;
};
