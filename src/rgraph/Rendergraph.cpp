#include "Rendergraph.h"
#include "GPUResourceAllocator.h"
#include "IFeature.h"
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

rgraph::Rendergraph *rgraph::Rendergraph::instance = nullptr;

void rgraph::Pass::ReadsImage(const std::string name, VkImageLayout layout)
{
    PassImageRead imageRead = {};
    imageRead.name = name;
    imageRead.startingLayout = layout;

    imageReads.emplace_back(std::move(imageRead));
}

void rgraph::Pass::WritesImage(const std::string name)
{
    PassImageWrite imageWrite = {};
    imageWrite.name = name;

    imageWrites.emplace_back(std::move(imageWrite));
}

void rgraph::Pass::AddColorAttachment(const std::string name, bool store, VkClearValue *clear)
{
    PassImageWrite imageWrite = {};
    imageWrite.clear = clear;
    imageWrite.store = store;
    imageWrite.name = name;

    colorAttachments.emplace_back(std::move(imageWrite));
}

void rgraph::Pass::AddColorAttachment(const std::string name, bool store, VkClearValue *clear, std::string resolveName,
                                      VkResolveModeFlagBits colorResolutionMode)
{
    // if store, write to color attachment, and if clearvalue is null, do not clear the value beforehand.
    PassImageWrite imageWrite = {};
    imageWrite.clear = clear;
    imageWrite.store = store;
    imageWrite.name = name;

    imageWrite.bResolve = true;
    imageWrite.resolveName = resolveName;
    imageWrite.resolutionMode = colorResolutionMode;

    colorAttachments.emplace_back(std::move(imageWrite));
}

void rgraph::Pass::AddDepthStencilAttachment(const std::string name, bool store, VkClearValue *clear)
{
    depthAttachment.clear = clear;
    depthAttachment.store = store;
    depthAttachment.name = name;
}

void rgraph::Pass::AddDepthStencilAttachment(const std::string name, bool store, VkClearValue *clear, std::string resolveName,
                                             VkResolveModeFlagBits depthResolutionMode)
{
    depthAttachment.clear = clear;
    depthAttachment.store = store;
    depthAttachment.name = name;

    depthAttachment.bResolve = true;
    depthAttachment.resolveName = resolveName;
    depthAttachment.resolutionMode = depthResolutionMode;
}

void rgraph::Rendergraph::AddComputePass(const std::string name, std::function<void(Pass &)> setup, std::function<void(PassExecution &)> run)
{
    rgraph::Pass pass;
    pass.type = PassType::Compute;
    pass.name = name;
    setup(pass);

    passData.emplace_back(std::move(pass));

    executionLambdas.emplace_back(std::move(run));
}

void rgraph::Rendergraph::AddGraphicsPass(const std::string name, std::function<void(Pass &)> setup, std::function<void(PassExecution &)> run)
{
    rgraph::Pass pass;
    pass.type = PassType::Graphics;
    pass.name = name;
    setup(pass);

    passData.emplace_back(std::move(pass));

    executionLambdas.emplace_back(run);
}

void rgraph::Rendergraph::AddTrackedImage(const std::string name, VkImageLayout startLayout, AllocatedImage image)
{
    // I dont know why the startLayout is required, so ignoring it for now.
    images[name] = image;
}

void rgraph::Rendergraph::AddTrackedBuffer(const std::string name, AllocatedBuffer buffer)
{
    buffers[name] = buffer;
}

void rgraph::Rendergraph::DeclareTransientImage(const std::string &name, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage)
{
    declaredImages[name] = {extent, format, usage};
}

// ---------------------------------------------------------------------------
// Memory aliasing helpers
// ---------------------------------------------------------------------------

void rgraph::Rendergraph::computeResourceLifetimes()
{
    resourceLifetimes.clear();
    for (auto &[name, _] : declaredImages)
    {
        resourceLifetimes[name] = {};
    }

    for (int i = 0; i < (int)passData.size(); i++)
    {
        const Pass &pass = passData[i];

        auto update = [&](const std::string &name)
        {
            auto it = resourceLifetimes.find(name);
            if (it == resourceLifetimes.end())
                return;
            it->second.firstPass = std::min(it->second.firstPass, i);
            it->second.lastPass = std::max(it->second.lastPass, i);
        };

        for (auto &r : pass.imageReads)
            update(r.name);
        for (auto &w : pass.imageWrites)
            update(w.name);
        for (auto &c : pass.colorAttachments)
        {
            update(c.name);
            if (c.bResolve)
                update(c.resolveName);
        }
        if (!pass.depthAttachment.name.empty())
            update(pass.depthAttachment.name);
    }
}

void rgraph::Rendergraph::destroyTransientImages()
{
    VmaAllocator vmaAllocator = GPUResourceAllocator::Instance().getAllocator();

    for (auto &[name, _] : declaredImages)
    {
        auto it = images.find(name);
        if (it == images.end())
            continue;
        AllocatedImage &img = it->second;
        if (img.imageView != VK_NULL_HANDLE)
            vkDestroyImageView(_device, img.imageView, nullptr);
        // Images created via vmaCreateAliasingImage are destroyed with vkDestroyImage;
        // the backing memory belongs to the pool, not the image.
        if (img.image != VK_NULL_HANDLE)
            vkDestroyImage(_device, img.image, nullptr);
        images.erase(it);
    }
    for (auto &pool : aliasPools)
        vmaFreeMemory(vmaAllocator, pool.allocation);
    aliasPools.clear();
}

void rgraph::Rendergraph::allocateTransientImages()
{
    VmaAllocator vmaAllocator = GPUResourceAllocator::Instance().getAllocator();

    // Collect images that have a valid (referenced) lifetime, sorted by firstPass so
    // the greedy algorithm assigns the earliest-starting images first.
    std::vector<std::string> sorted;
    sorted.reserve(declaredImages.size());
    for (auto &[name, _] : declaredImages)
    {
        if (resourceLifetimes[name].valid())
            sorted.push_back(name);
    }
    std::sort(sorted.begin(), sorted.end(),
              [&](const std::string &a, const std::string &b)
              { return resourceLifetimes[a].firstPass < resourceLifetimes[b].firstPass; });

    for (auto &name : sorted)
    {
        const ImageDecl &decl = declaredImages[name];
        const ResourceLifetime &lifetime = resourceLifetimes[name];

        // VK_IMAGE_CREATE_ALIAS_BIT is required on every image that may share memory
        // with another image
        VkImageCreateInfo imgInfo = vkinit::image_create_info(decl.format, decl.usage, decl.extent);
        imgInfo.flags |= VK_IMAGE_CREATE_ALIAS_BIT;

        // Create a probe image (not yet bound to any memory) purely to query requirements.
        VkImage probeImage;
        VK_CHECK(vkCreateImage(_device, &imgInfo, nullptr, &probeImage));
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(_device, probeImage, &memReqs);

        // Greedy search: find a pool that is free at firstPass and large enough.
        AliasPool *chosen = nullptr;
        for (auto &pool : aliasPools)
        {
            bool isFree = pool.releaseAfterPass < lifetime.firstPass;
            bool fitsSize = pool.size >= memReqs.size;
            bool compatibleType = (pool.memoryTypeBit & memReqs.memoryTypeBits) != 0;
            if (isFree && fitsSize && compatibleType)
            {
                chosen = &pool;
                break;
            }
        }

        VkImage image;
        if (chosen)
        {
            // Alias into an existing pool: probe is no longer needed.
            vkDestroyImage(_device, probeImage, nullptr);
            VK_CHECK(vmaCreateAliasingImage(vmaAllocator, chosen->allocation, &imgInfo, &image));
            chosen->releaseAfterPass = lifetime.lastPass;
        }
        else
        {
            // No compatible free pool — allocate a new VmaAllocation sized for this image,
            // then alias the real image into it and discard the probe.
            VmaAllocationCreateInfo allocCI = {};
            allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            VmaAllocation newAlloc;
            VmaAllocationInfo allocInfo;
            VK_CHECK(vmaAllocateMemoryForImage(vmaAllocator, probeImage, &allocCI, &newAlloc, &allocInfo));

            vkDestroyImage(_device, probeImage, nullptr);
            VK_CHECK(vmaCreateAliasingImage(vmaAllocator, newAlloc, &imgInfo, &image));

            aliasPools.push_back({newAlloc, memReqs.size, 1u << allocInfo.memoryType, lifetime.lastPass});
        }

        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        if (decl.format == VK_FORMAT_D32_SFLOAT || decl.format == VK_FORMAT_D24_UNORM_S8_UINT ||
            decl.format == VK_FORMAT_D16_UNORM || decl.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        {
            aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        }

        VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(decl.format, image, aspect);
        VkImageView imageView;
        VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &imageView));

        AllocatedImage allocImg = {};
        allocImg.image = image;
        allocImg.imageView = imageView;
        allocImg.allocation = VK_NULL_HANDLE; // memory is owned by the AliasPool, not this image
        allocImg.imageExtent = decl.extent;
        allocImg.imageFormat = decl.format;

        images[name] = allocImg;
    }
}

void rgraph::Rendergraph::Cleanup()
{
    destroyTransientImages();
    transientImagesAllocated = false;
}

// ---------------------------------------------------------------------------

void rgraph::Rendergraph::Build(FrameData &frameData)
{
    transitionData.clear();
    passData.clear();
    executionLambdas.clear();
    // buffers.clear();
    // need to insert image transitions between the features.

    // so this loops through the different required IFeatures, then calls the setup lambdas, then finally inserts
    // the transitions.
    /// currently this depends on the order in which features are added.
    for (auto &feature : features)
    {
        feature.lock()->Register(this);
    }

    // all the AddXPass would be called above.

    // Compute per-resource lifetimes from the just-populated passData.
    // Allocate transient images (with memory aliasing) on the first Build only.
    if (!declaredImages.empty())
    {
        computeResourceLifetimes();
        if (!transientImagesAllocated)
        {
            allocateTransientImages();
            transientImagesAllocated = true;
        }
    }

    std::unordered_map<std::string, VkImageLayout> imgLayoutMap;

    for (auto &image : images)
    {
        imgLayoutMap[image.first] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    for (auto &pass : passData)
    {
        transitionData[pass.name] = {};
        // get the image writes.
        for (auto &writeImage : pass.imageWrites)
        {
            // all these will be in general.
            if (imgLayoutMap[writeImage.name] != VK_IMAGE_LAYOUT_GENERAL)
            {
                // add a transition here.
                transitionData[pass.name].push_back({writeImage.name, imgLayoutMap[writeImage.name], VK_IMAGE_LAYOUT_GENERAL});
                imgLayoutMap[writeImage.name] = VK_IMAGE_LAYOUT_GENERAL;
            }
        }

        // get the image reads.
        for (auto &readImage : pass.imageReads)
        {
            // need to check if current layout and new layout are same
            if (imgLayoutMap[readImage.name] != readImage.startingLayout)
            {
                // add a transition here.
                transitionData[pass.name].push_back({readImage.name, imgLayoutMap[readImage.name], readImage.startingLayout});
                imgLayoutMap[readImage.name] = readImage.startingLayout;
            }
        }

        // get the color and depth attachments later, they should be in color/depth _optimal
        for (auto &colorImage : pass.colorAttachments)
        {
            if (imgLayoutMap[colorImage.name] != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
            {
                // transition to color layout here.
                transitionData[pass.name].push_back({colorImage.name, imgLayoutMap[colorImage.name], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
                imgLayoutMap[colorImage.name] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }

            // resolve targets as well
            if (colorImage.bResolve)
            {
                if (imgLayoutMap[colorImage.resolveName] != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                {
                    transitionData[pass.name].push_back(
                        {colorImage.resolveName, imgLayoutMap[colorImage.resolveName], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
                    imgLayoutMap[colorImage.resolveName] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                }
            }
        }

        // similarly, now do the same for depth image. Don't need a loop because the depth image will always be
        // singular.
        if (imgLayoutMap.contains(pass.depthAttachment.name) && imgLayoutMap[pass.depthAttachment.name] != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        {
            // transition data.
            transitionData[pass.name].push_back(
                {pass.depthAttachment.name, imgLayoutMap[pass.depthAttachment.name], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL});
            imgLayoutMap[pass.depthAttachment.name] = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

            // also do resolve targets
            if (pass.depthAttachment.bResolve)
            {
                if (imgLayoutMap[pass.depthAttachment.resolveName] != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                {
                    transitionData[pass.name].push_back(
                        {pass.depthAttachment.resolveName, imgLayoutMap[pass.depthAttachment.resolveName], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL});
                    imgLayoutMap[pass.depthAttachment.resolveName] = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                }
            }
        }
    }
}

void rgraph::Rendergraph::Run(FrameData &frameData)
{
    // actually call the execution lambdas here.
    // Ideally this should already contain the transition stuff.

    auto cpuTimeStart = std::chrono::system_clock::now();
    VkCommandBuffer cmd = frameData._mainCommandBuffer;
    VkQueryPool queryPool = frameData.timestampQueryPool;

    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    // start command buffer recording ---------------------
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    uint32_t timestampCount = passData.size() * 2 + 2;
    vkCmdResetQueryPool(cmd, queryPool, 0, timestampCount);

    std::vector<std::pair<std::string, uint32_t>> passIndices;
    uint32_t queryIndex = 0;

    uint32_t totalStartQuery = queryIndex++;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, totalStartQuery);

    for (size_t i = 0; i < passData.size(); i++)
    {

        auto passStartTime = std::chrono::system_clock::now();

        const Pass &pass = passData[i];

        VkDebugUtilsLabelEXT label = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = pass.name.c_str();
        vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, queryIndex);
        uint32_t startQuery = queryIndex++;

        // Insert transitions for this pass
        auto transitionsIt = transitionData.find(pass.name);
        if (transitionsIt != transitionData.end())
        {
            for (const auto &transition : transitionsIt->second)
            {
                AllocatedImage img = images[transition.imageName];
                barrierMerger.transition_image(img.image, transition.currentLayout, transition.newLayout);
            }
        }

        barrierMerger.flushBarriers(frameData._mainCommandBuffer);

        // Create buffers if required.
        PassExecution exec;

        GPUResourceAllocator *gpuResourceAllocator = &GPUResourceAllocator::Instance();

        // potential for memory aliasing here.
        if (pass.bufferCreations.size() > 0)
        {
            // create the buffers.
            for (auto &bufferCreateInfo : pass.bufferCreations)
            {
                AllocatedBuffer newBuffer =
                    gpuResourceAllocator->create_buffer(bufferCreateInfo.size, bufferCreateInfo.usageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU);
                exec.allocatedBuffers[bufferCreateInfo.name] = newBuffer;
                frameData._deletionQueue.push_function([=, this]() { gpuResourceAllocator->destroy_buffer(newBuffer); });
            }
        }

        // Create unique PassExecution for this pass
        // PassExecution exec;
        exec.cmd = frameData._mainCommandBuffer;
        exec._device = _device;
        exec._drawExtent = _extent;
        exec.allocatedImages = images;
        exec.delQueue = &(frameData._deletionQueue);
        exec.frameDescriptor = &(frameData._frameDescriptors);

        // Execute the pass with its own context
        if (pass.type == PassType::Graphics)
        {

            std::vector<VkRenderingAttachmentInfo> colorAttachments;
            AllocatedImage depthImage = images[pass.depthAttachment.name];
            for (auto &colAttachment : pass.colorAttachments)
            {
                AllocatedImage drawImage = images[colAttachment.name];
                VkRenderingAttachmentInfo colorAttachment;
                if (!colAttachment.bResolve)
                {
                    colorAttachment = vkinit::attachment_info(drawImage.imageView, colAttachment.clear, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                }
                else
                {
                    colorAttachment = vkinit::attachment_info(drawImage.imageView, colAttachment.clear, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                              images[colAttachment.resolveName].imageView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                              colAttachment.resolutionMode);
                }
                colorAttachments.emplace_back(std::move(colorAttachment));
            }

            VkRenderingAttachmentInfo depthAttachment;

            // setup depth attachment similarly
            if (!pass.depthAttachment.bResolve)
            {
                depthAttachment =
                    vkinit::depth_attachment_info(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, pass.depthAttachment.clear);
            }
            else
            {
                depthAttachment = vkinit::depth_attachment_info(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                                                images[pass.depthAttachment.resolveName].imageView,
                                                                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, pass.depthAttachment.resolutionMode);
            }
            VkRenderingInfo renderInfo = vkinit::rendering_info({_extent.width, _extent.height}, &colorAttachments, &depthAttachment);
            vkCmdBeginRendering(cmd, &renderInfo);
        }
        executionLambdas[i](exec);

        if (pass.type == PassType::Graphics)
        {
            vkCmdEndRendering(cmd);
        }

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, queryIndex);
        queryIndex++;

        vkCmdEndDebugUtilsLabelEXT(cmd);

        // save timestamps for time queries later.
        passIndices.push_back({pass.name, startQuery});

        auto passEndTime = std::chrono::system_clock::now();
        auto passTime = std::chrono::duration_cast<std::chrono::microseconds>(passEndTime - passStartTime);
        PassStats stats;
        stats.name = pass.name;
        if (pass.type == PassType::Compute)
        {
            stats.computeDispatches = exec.dispatchCalls;
        }
        else
        {
            stats.draws = exec.drawCalls;
            stats.triangles = exec.triangles;
        }
        stats.CPUTime = passTime.count() / 1000.0f;
        frameData.stats.passStats.push_back(stats);
    }

    uint32_t totalEndQuery = queryIndex++;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, totalEndQuery);

    frameData.timestampCount = timestampCount;
    frameData.passIndices = passIndices;
    frameData.totalTimeIndices = {totalStartQuery, totalEndQuery};
    // commenting this out for now, will change later
    // TODO: move swapchain transitions into the rendergraph.
    // VK_CHECK(vkEndCommandBuffer(cmd));

    auto cpuEndTime = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(cpuEndTime - cpuTimeStart);
    frameData.stats.CPUFrametime = elapsed.count() / 1000.f;
}

void rgraph::Rendergraph::AddFeature(std::weak_ptr<IFeature> feature)
{
    features.emplace_back(feature);
}

void rgraph::Rendergraph::Init(VkDevice _device, VkExtent3D _extent, VkInstance _instance)
{
    if (instance != nullptr)
    {
        assert(true && "rendergraph has already been initialized");
    }
    this->_device = _device;
    this->_extent = _extent;
    instance = this;

    vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(_instance, "vkCmdBeginDebugUtilsLabelEXT");
    vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(_instance, "vkCmdEndDebugUtilsLabelEXT");
}

rgraph::Rendergraph &rgraph::Rendergraph::Instance()
{
    if (instance)
    {
        return *instance;
    }
    throw(true && "Rendergraph has not been initialized yet!");
}

void rgraph::Pass::CreatesBuffer(const std::string name, size_t size, VkBufferUsageFlags usages)
{
    PassBufferCreationInfo bufferCreateInfo = {};
    bufferCreateInfo.name = name;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usageFlags = usages;

    bufferCreations.emplace_back(bufferCreateInfo);
}

void rgraph::Pass::ReadsBuffer(const std::string name)
{
    // do nothing right now, not sure where these are used yet.
}

void rgraph::Pass::WritesBuffer(const std::string name)
{
    // do nothing right now, not sure where these are used yet.
}

void rgraph::Rendergraph::ReadTimestamps(FrameData &frameData)
{
    if (frameData.timestampCount == 0)
    {
        return;
    }

    std::vector<uint64_t> timestamps(frameData.timestampCount);

    VkResult result = vkGetQueryPoolResults(_device, frameData.timestampQueryPool, 0, frameData.timestampCount, timestamps.size() * sizeof(uint64_t),
                                            timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (result != VK_SUCCESS)
    {
        return;
    }

    for (size_t i = 0; i < frameData.passIndices.size() && i < frameData.stats.passStats.size(); i++)
    {
        uint32_t startIdx = frameData.passIndices[i].second;
        uint64_t start = timestamps[startIdx];
        uint64_t end = timestamps[startIdx + 1];
        uint64_t duration = (end >= start) ? (end - start) : (UINT64_MAX - start + end);

        frameData.stats.passStats[i].GPUTime = duration * timestampPeriod / 1000000.0f;
    }

    uint64_t totalStart = timestamps[frameData.totalTimeIndices.first];
    uint64_t totalEnd = timestamps[frameData.totalTimeIndices.second];
    uint64_t totalDuration = (totalEnd >= totalStart) ? (totalEnd - totalStart) : (UINT64_MAX - totalStart + totalEnd);
    frameData.stats.totalGPUTime = totalDuration * timestampPeriod / 1000000.0f;
}