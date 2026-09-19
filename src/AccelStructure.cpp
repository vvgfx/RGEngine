#include "AccelStructure.h"
#include "fmt/base.h"
#include "GPUResourceAllocator.h"
#include "vk_engine.h"

namespace
{
    // The engine links the static Vulkan loader, so KHR ray-tracing entry points must be fetched by hand.
    // Same pattern as the debug-label pointers in Rendergraph.
    struct RtApi
    {
        PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
        PFN_vkCreateAccelerationStructureKHR create = nullptr;
        PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR getAddress = nullptr;

        bool Load(VkDevice device)
        {
            getBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
            create = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
            destroy = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
            cmdBuild = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
            getAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
            return getBuildSizes && create && destroy && cmdBuild && getAddress;
        }
    };

    RtApi rt;

    VkDeviceAddress bufferAddress(VkDevice device, VkBuffer buffer)
    {
        VkBufferDeviceAddressInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer};
        return vkGetBufferDeviceAddress(device, &info);
    }

    constexpr VkBufferUsageFlags AS_STORAGE = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr VkBufferUsageFlags SCRATCH = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
} // namespace

bool AccelStructure::Build(VulkanEngine *engine, VkDevice device, const DrawContext &ctx, DeletionQueue &delQueue)
{
    if (!rt.Load(device))
    {
        fmt::println("AccelStructure: ray tracing entry points unavailable");
        return false;
    }

    const auto &objects = ctx.OpaqueSurfaces;
    if (objects.empty())
    {
        return false;
    }

    auto &alloc = GPUResourceAllocator::Instance();

    // Scratch offsets have a device-specific alignment requirement.
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &asProps};
    vkGetPhysicalDeviceProperties2(engine->GetPhysicalDevice(), &props2);
    const VkDeviceSize scratchAlign = asProps.minAccelerationStructureScratchOffsetAlignment;

    const size_t count = objects.size();

    // --- one BLAS per submesh; batched into a single build command ---
    std::vector<VkAccelerationStructureGeometryKHR> geometries(count);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(count);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(count);
    std::vector<VkDeviceSize> scratchOffsets(count);

    blases.resize(count);
    VkDeviceSize scratchTotal = 0;

    for (size_t i = 0; i < count; i++)
    {
        const RenderObject &obj = objects[i];

        geometries[i] = VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry = {.triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                                       .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                                       .vertexData = {.deviceAddress = obj.vertexBufferAddress},
                                       .vertexStride = sizeof(Vertex),
                                       .maxVertex = obj.vertexCount > 0 ? obj.vertexCount - 1 : 0,
                                       .indexType = VK_INDEX_TYPE_UINT32,
                                       .indexData = {.deviceAddress = obj.indexBufferAddress}}},
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};

        const uint32_t primitiveCount = obj.indexCount / 3;
        ranges[i] = VkAccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = primitiveCount, .primitiveOffset = obj.firstIndex * uint32_t(sizeof(uint32_t)), .firstVertex = 0, .transformOffset = 0};

        buildInfos[i] = VkAccelerationStructureBuildGeometryInfoKHR{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                                                                    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                                                    .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                                                                    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                                                                    .geometryCount = 1,
                                                                    .pGeometries = &geometries[i]};

        VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        rt.getBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfos[i], &primitiveCount, &sizes);

        blases[i].buffer = alloc.create_buffer(sizes.accelerationStructureSize, AS_STORAGE, VMA_MEMORY_USAGE_GPU_ONLY);

        VkAccelerationStructureCreateInfoKHR createInfo{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                                        .buffer = blases[i].buffer.buffer,
                                                        .size = sizes.accelerationStructureSize,
                                                        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
        VK_CHECK(rt.create(device, &createInfo, nullptr, &blases[i].handle));

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                                             .accelerationStructure = blases[i].handle};
        blases[i].address = rt.getAddress(device, &addrInfo);

        buildInfos[i].dstAccelerationStructure = blases[i].handle;

        scratchOffsets[i] = scratchTotal;
        scratchTotal += (sizes.buildScratchSize + scratchAlign - 1) & ~(scratchAlign - 1);
    }

    AllocatedBuffer blasScratch = alloc.create_buffer(scratchTotal, SCRATCH, VMA_MEMORY_USAGE_GPU_ONLY);
    const VkDeviceAddress blasScratchAddr = bufferAddress(device, blasScratch.buffer);

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> rangePtrs(count);
    for (size_t i = 0; i < count; i++)
    {
        buildInfos[i].scratchData.deviceAddress = blasScratchAddr + scratchOffsets[i];
        rangePtrs[i] = &ranges[i];
    }

    // --- TLAS instances + the geometry table the hit shader reads ---
    AllocatedBuffer instanceBuffer = alloc.create_buffer(sizeof(VkAccelerationStructureInstanceKHR) * count,
                                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                         VMA_MEMORY_USAGE_CPU_TO_GPU);

    geometryBufferSize = sizeof(GPUGeometryInfo) * count;
    geometryBuffer = alloc.create_buffer(geometryBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto *instances = (VkAccelerationStructureInstanceKHR *)instanceBuffer.info.pMappedData;
    auto *geoInfos = (GPUGeometryInfo *)geometryBuffer.info.pMappedData;

    for (size_t i = 0; i < count; i++)
    {
        const RenderObject &obj = objects[i];

        // VkTransformMatrixKHR is a row-major 3x4; glm is column-major, so transpose while copying.
        VkTransformMatrixKHR transform{};
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                transform.matrix[row][col] = obj.modelMatrix[col][row];
            }
        }

        instances[i] = VkAccelerationStructureInstanceKHR{.transform = transform,
                                                          .instanceCustomIndex = uint32_t(i),
                                                          .mask = 0xFF,
                                                          .instanceShaderBindingTableRecordOffset = 0,
                                                          .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                                                          .accelerationStructureReference = blases[i].address};

        geoInfos[i] = GPUGeometryInfo{.vertexAddress = obj.vertexBufferAddress,
                                      .indexAddress = obj.indexBufferAddress,
                                      .colorFactor = obj.material ? obj.material->colorFactor : glm::vec4(1.0f),
                                      .firstIndex = obj.firstIndex,
                                      .albedoTexIndex = obj.material ? obj.material->albedoTexIndex : 0u};
    }

    VkAccelerationStructureGeometryKHR tlasGeometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                   .arrayOfPointers = VK_FALSE,
                                   .data = {.deviceAddress = bufferAddress(device, instanceBuffer.buffer)}}},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                                                          .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                                          .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                                                          .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                                                          .geometryCount = 1,
                                                          .pGeometries = &tlasGeometry};

    const uint32_t instanceCount = uint32_t(count);
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    rt.getBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild, &instanceCount, &tlasSizes);

    tlasBuffer = alloc.create_buffer(tlasSizes.accelerationStructureSize, AS_STORAGE, VMA_MEMORY_USAGE_GPU_ONLY);

    VkAccelerationStructureCreateInfoKHR tlasCreate{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                                    .buffer = tlasBuffer.buffer,
                                                    .size = tlasSizes.accelerationStructureSize,
                                                    .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    VK_CHECK(rt.create(device, &tlasCreate, nullptr, &tlas));

    AllocatedBuffer tlasScratch = alloc.create_buffer(tlasSizes.buildScratchSize, SCRATCH, VMA_MEMORY_USAGE_GPU_ONLY);
    tlasBuild.dstAccelerationStructure = tlas;
    tlasBuild.scratchData.deviceAddress = bufferAddress(device, tlasScratch.buffer);

    VkAccelerationStructureBuildRangeInfoKHR tlasRange{.primitiveCount = instanceCount};
    const VkAccelerationStructureBuildRangeInfoKHR *tlasRangePtr = &tlasRange;

    engine->immediate_submit(
        [&](VkCommandBuffer cmd)
        {
            rt.cmdBuild(cmd, uint32_t(count), buildInfos.data(), rangePtrs.data());

            // the TLAS build reads the BLASes just written.
            VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                     .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                     .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
            VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
            vkCmdPipelineBarrier2(cmd, &dep);

            rt.cmdBuild(cmd, 1, &tlasBuild, &tlasRangePtr);
        });

    // immediate_submit is fully synchronous, so scratch and instance staging are dead by now.
    alloc.destroy_buffer(blasScratch);
    alloc.destroy_buffer(tlasScratch);
    alloc.destroy_buffer(instanceBuffer);

    delQueue.push_function(
        [device, this]()
        {
            auto &a = GPUResourceAllocator::Instance();
            rt.destroy(device, tlas, nullptr);
            a.destroy_buffer(tlasBuffer);
            a.destroy_buffer(geometryBuffer);
            for (auto &b : blases)
            {
                rt.destroy(device, b.handle, nullptr);
                a.destroy_buffer(b.buffer);
            }
        });

    fmt::println("AccelStructure: built {} BLAS + TLAS", count);
    return true;
}
