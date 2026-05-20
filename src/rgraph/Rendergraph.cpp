#include "Rendergraph.h"
#include "GPUResourceAllocator.h"
#include <fmt/core.h>
#include "IFeature.h"
#include "vk_engine.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <cassert>
#include <chrono>
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
    if (name.empty()) {
        fmt::println("AddTrackedImage called with empty name, image={}", (void*)image.image);
    }
    images[name] = image;
}

void rgraph::Rendergraph::AddTrackedBuffer(const std::string name, AllocatedBuffer buffer)
{
    buffers[name] = buffer;
}

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
        if (!pass.depthAttachment.name.empty() && imgLayoutMap.contains(pass.depthAttachment.name) && imgLayoutMap[pass.depthAttachment.name] != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
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
                if (!img.image) {
                    fmt::println("NULL image '{}' in pass '{}'", transition.imageName, pass.name);
                }
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