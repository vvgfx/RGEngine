#pragma once
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_types.h"
#include <climits>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
namespace rgraph
{

    class IFeature;

    enum PassType
    {
        Compute,
        Graphics
    };

    struct PassImageRead
    {
        std::string name;
        VkImageLayout startingLayout;
    };

    struct PassImageWrite
    {
        std::string name;

        // These are used for color and depth attachments
        bool store;
        VkClearValue *clear = nullptr;

        // for resolution
        bool bResolve = false;
        std::string resolveName;
        VkResolveModeFlagBits resolutionMode;
    };

    struct PassBufferCreationInfo
    {
        std::string name;
        size_t size;
        VkBufferUsageFlags usageFlags;
    };

    struct Pass
    {
        friend class Rendergraph;

        // these are for compute.
        void ReadsImage(const std::string name, VkImageLayout layout);
        void WritesImage(const std::string name);

        // these are for graphics.
        void AddColorAttachment(const std::string name, bool store, VkClearValue *clear = nullptr);

        // default colorResolutionMode = VK_RESOLVE_MODE_AVERAGE_BIT. Pass nullptr to clear to keep value.
        void AddColorAttachment(const std::string name, bool store, VkClearValue *clear, std::string resolveName,
                                VkResolveModeFlagBits colorResolutionMode = VK_RESOLVE_MODE_AVERAGE_BIT);

        void AddDepthStencilAttachment(const std::string name, bool store, VkClearValue *clear = nullptr);

        // default depthResolutionMode = VK_RESOLVE_MODE_MAX_BIT (using reverse-z, so MAX, not MIN). Pass nullptr to
        // clear to keep value.
        void AddDepthStencilAttachment(const std::string name, bool store, VkClearValue *clear, std::string resolveName,
                                       VkResolveModeFlagBits depthResolutionMode = VK_RESOLVE_MODE_MAX_BIT);

        void CreatesBuffer(const std::string name, size_t size, VkBufferUsageFlags usages);

        void ReadsBuffer(const std::string name);
        void WritesBuffer(const std::string name);

        PassType type;
        std::string name;

      private:
        std::vector<PassImageRead> imageReads;
        std::vector<PassImageWrite> imageWrites;

        std::vector<PassImageWrite> colorAttachments;

        std::vector<PassBufferCreationInfo> bufferCreations;
        // add string vector for buffer dependencies

        // add depth attachment read, bool storeDepth, and a reference to the creating builder itself.
        PassImageWrite depthAttachment{};
        bool storeDepth;
    };

    struct PassExecution
    {
        VkCommandBuffer cmd;
        VkDevice _device;
        std::unordered_map<std::string, AllocatedBuffer> allocatedBuffers;
        std::unordered_map<std::string, AllocatedImage> allocatedImages;

        // temporary, need to change later.
        VkExtent3D _drawExtent;

        // stuff required for per-frame data.
        DeletionQueue *delQueue;
        DescriptorAllocatorGrowable *frameDescriptor;

        // performance variables.
        float dispatchCalls; // compute
        float drawCalls;     // graphics
        float triangles;     // graphics
    };

    struct TransitionData
    {
        std::string imageName;
        VkImageLayout currentLayout, newLayout;
    };

    struct PassTiming
    {
        std::string name;
        float gpuMs = 0.0f;
        uint32_t barrierCount = 0;
    };

    // Description of a transient image to be allocated with memory aliasing.
    struct ImageDecl
    {
        VkExtent3D extent;
        VkFormat format;
        VkImageUsageFlags usage;
    };

    // First and last pass index (inclusive) in which a resource is accessed.
    struct ResourceLifetime
    {
        int firstPass = INT_MAX;
        int lastPass = -1;

        bool valid() const { return lastPass >= 0; }

        // Two lifetimes overlap when either starts before the other ends.
        bool overlaps(const ResourceLifetime &other) const
        {
            return firstPass <= other.lastPass && other.firstPass <= lastPass;
        }
    };

    /**
     * @brief This class builds the rendergraph, and is expected to be called every frame.
     *
     */
    class Rendergraph
    {

      public:
        void AddComputePass(const std::string name, std::function<void(Pass &)> setup, std::function<void(PassExecution &)> run);
        void AddGraphicsPass(const std::string name, std::function<void(Pass &)> setup, std::function<void(PassExecution &)> run);

        void AddTrackedImage(const std::string name, VkImageLayout startLayout, AllocatedImage image);
        void AddTrackedBuffer(const std::string name, AllocatedBuffer buffer);

        // Declare a transient image that the rendergraph owns and will allocate with memory aliasing.
        // Call this once per image (e.g. in feature constructors) before the first Build().
        void DeclareTransientImage(const std::string &name, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage);

        void Build(FrameData &frameData);

        // the framedata is used for per-frame deletion queue and unique command buffers.
        void Run(FrameData &frameData);

        void AddFeature(std::weak_ptr<IFeature> feature);

        void Init(VkDevice _device, VkExtent3D _extent, VkInstance _instance);

        // Destroy all transient images and free aliased memory. Call once at shutdown.
        void Cleanup();

        // performance stuff.
        void SetTimestampPeriod(float period)
        {
            timestampPeriod = period;
        }
        const std::vector<PassTiming> &GetLastFrameTimings() const
        {
            return lastFrameTimings;
        }
        void ReadTimestamps(FrameData &frameData);

        float GetTotalGpuTime() const
        {
            return totalGpuMs;
        }

        static Rendergraph &Instance();

      private:
        std::vector<Pass> passData;

        std::vector<std::function<void(PassExecution &)>> executionLambdas;

        std::unordered_map<std::string, AllocatedImage> images;
        std::unordered_map<std::string, AllocatedBuffer> buffers;

        std::vector<std::weak_ptr<IFeature>> features;

        std::unordered_map<std::string, std::vector<TransitionData>> transitionData;
        VkDevice _device;
        VkExtent3D _extent;

        // performance stuff.
        std::vector<PassTiming> lastFrameTimings;
        std::vector<uint64_t> timestampBuffer;
        float timestampPeriod = 1.0f;
        float totalGpuMs = 0.0f;

        vkutil::BarrierMerger barrierMerger;

        static Rendergraph *instance;

        // https://github.com/achirkin/vulkan/issues/3
        // I really don't like doing this, but it seems to be the only way?

        PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT = nullptr;
        PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT = nullptr;

        // ---- Memory aliasing ----

        // A single VMA allocation that may be shared by multiple non-overlapping transient images.
        struct AliasPool
        {
            VmaAllocation allocation;
            VkDeviceSize size;
            uint32_t memoryTypeBit; // (1u << memoryType) for compatibility checks
            int releaseAfterPass;   // last pass of the current owner; -1 when never assigned
        };

        std::unordered_map<std::string, ImageDecl> declaredImages;
        std::unordered_map<std::string, ResourceLifetime> resourceLifetimes;
        std::vector<AliasPool> aliasPools;
        bool transientImagesAllocated = false;

        // Scan passData to fill resourceLifetimes for every declared image.
        void computeResourceLifetimes();

        // Greedy interval-based allocator: assigns each declared image to an alias pool
        // and binds the VkImage to its pool's VkDeviceMemory, then inserts the result
        // into the images map so the rest of the graph sees it like any other image.
        void allocateTransientImages();

        // Destroy VkImages / VkImageViews owned by the aliasing system and free all pools.
        void destroyTransientImages();
    };
} // namespace rgraph