#pragma once

#include "sgraph/ScenegraphStructs.h"
#include "vk_descriptors.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vk_types.h>

// importing PBEngine gives me circular dependency issues,so forward declaring this for now.
struct MaterialSystem;

// This holds the material data for one submesh (GeoSurface) (Color image, metallic-roughness image, AllocatedBuffer
// pointer and offets for the material constants like color-factor and metallic-roughness-factor)
struct GLTFMaterial
{
    MaterialInstance data;
};

// Bounds - used for frustum culling
struct Bounds
{
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

// Geometry-Surface - This is a submesh. Each GeoSurface corresponds to one drawcall.
struct GeoSurface
{
    uint32_t startIndex;
    uint32_t count;
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

// This is a singular mesh; it contains a vector of GeoSurfaces (submeshes) and has it's own GPUMeshBuffer
struct MeshAsset
{
    std::string name;
    std::vector<GeoSurface> surfaces; // submesh
    GPUMeshBuffers meshBuffers;

    // GPU buffers above are write-only; physics builds hulls from this. Mesh-local.
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

// contains details required for the loaders.
struct GLTFCreatorData
{
    VkDevice _device;
    AllocatedImage loadErrorImage;
    AllocatedImage defaultImage;
    VkSampler _defaultSamplerLinear;
    MaterialSystem *materialSystemReference;
};

// lighting data
struct LightingData
{
    glm::vec3 color;
    float intensity;
    enum LightType : uint8_t
    {
        Directional,
        Spot,
        Point
    };

    LightType type;
    float range;
    float innerConeAngle;
    float outerConeAngle;
    std::string name;
};

class VulkanEngine;

namespace sgraph
{

    /**
     * This class has a 1:1 relation with a GLTF file. The idea is that it is self-contained. It holds its' own data,
     * and all the data is cleaned up when the node is destroyed.
     */
    struct Scene : public INode
    {
        // storage for all the data on a given glTF file

        std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
        std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
        std::unordered_map<std::string, AllocatedImage> images;
        std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;
        std::unordered_map<std::string, std::shared_ptr<LightingData>> lightingData;

        // nodes that dont have a parent, for iterating through the file in tree order
        std::vector<std::shared_ptr<Node>> topNodes;

        std::vector<VkSampler> samplers;

        // Each material requires a descriptor set that knows which offset on which buffer contains the material data
        // like metallic-roughness colors and colorFactors; It also needs to know the images and samplers for colors and
        // metallicRoughness. This data is saved in GLTFMaterial (materials) and the buffer points to
        // the materialDataBuffer of that GLTFScene.

        // Each GLTF file has it's own descriptor pool. The descriptor sets allocated from this pool hold
        // GLTFMRMaterialSystem::MaterialResources (This contains the material's images, samplers, and pointers and
        // offsets to the AllocatedBuffer for this scene's material constants.)
        DescriptorAllocatorGrowable descriptorPool;

        // This holds all the GLTFMRMaterialSystem::MaterialConstants ( colorFactors, metal_rough_factors ) data for all
        // the materials in the scene packed together.
        AllocatedBuffer materialDataBuffer;

        GLTFCreatorData creator;

        ~Scene()
        {
            clearAll();
        };

        virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);

        std::string name;

      private:
        void clearAll();
    };

} // namespace sgraph

std::optional<std::shared_ptr<sgraph::Scene>> loadGltf(GLTFCreatorData creatorData, std::string_view filePath);

std::optional<std::shared_ptr<AllocatedImage>> loadImage(std::string fileName);