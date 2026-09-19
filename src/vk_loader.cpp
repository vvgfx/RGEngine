#include "GPUResourceAllocator.h"
#include "dds_loader.h"
#include "MaterialSystem.h"
#include "fastgltf/types.hpp"
#include "fmt/base.h"
#include "sgraph/ScenegraphStructs.h"
#include "stb_image.h"
#include "vk_engine.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vk_loader.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <vulkan/vulkan_core.h>

// forward declaration of global functions
VkFilter extract_filter(fastgltf::Filter filter);
VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter);
std::optional<AllocatedImage> load_image(fastgltf::Asset &asset, fastgltf::Image &image, const std::filesystem::path &baseDir);

std::optional<std::shared_ptr<sgraph::Scene>> loadGltf(std::string_view filePath)
{
    fmt::print("Loading GLTF: {}", filePath);

    VulkanEngine &engine = VulkanEngine::Instance();
    VkDevice device = engine.GetVkDevice();

    std::shared_ptr<sgraph::Scene> scene = std::make_shared<sgraph::Scene>();
    sgraph::Scene &file = *scene.get();

    // MSFT_texture_dds must be enabled or fastgltf leaves Texture::ddsImageIndex empty and a
    // DDS-backed glTF resolves every texture to a PNG fallback that may not ship. Kept so the old
    // zeux asset still loads. emissive_strength matters here: Bistro's emissive factors reach 100.
    fastgltf::Parser parser(fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::KHR_materials_emissive_strength |
                            fastgltf::Extensions::MSFT_texture_dds);

    constexpr auto gltfOptions =
        fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadExternalBuffers;

    auto dataResult = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!dataResult)
    {
        fmt::print("Failed to load file: {} \n", fastgltf::to_underlying(dataResult.error()));
        return {};
    }

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(dataResult.get());
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGltf(dataResult.get(), path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadGltfBinary(dataResult.get(), path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else
    {
        std::cerr << "Failed to determine glTF container" << std::endl;
        return {};
    }

    // we can estimate the descriptors we will need accurately
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};

    file.descriptorPool.init(device, gltf.materials.size(), sizes);

    // load samplers
    for (fastgltf::Sampler &sampler : gltf.samplers)
    {

        VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr};
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;

        // glTF samplers routinely omit filters; nearest is a poor default, linear is what content expects.
        sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear));
        sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Linear));

        sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::LinearMipMapLinear));

        // anisotropy is only legal when both filters are LINEAR
        if (sampl.magFilter == VK_FILTER_LINEAR && sampl.minFilter == VK_FILTER_LINEAR)
        {
            sampl.anisotropyEnable = VK_TRUE;
            sampl.maxAnisotropy = 16.f;
        }

        VkSampler newSampler;
        vkCreateSampler(device, &sampl, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }

    // temporal arrays for all the objects to use while creating the GLTF data
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<sgraph::Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;
    std::vector<std::shared_ptr<LightingData>> lights;

    // load all lights first
    for (fastgltf::Light &light : gltf.lights)
    {
        std::shared_ptr<LightingData> ldata = std::make_shared<LightingData>();

        ldata->color = glm::vec3(light.color[0], light.color[1], light.color[2]);
        ldata->intensity = light.intensity;
        ldata->range = light.range.value_or(-1);
        ldata->innerConeAngle = light.innerConeAngle.value_or(-1);
        ldata->outerConeAngle = light.outerConeAngle.value_or(-1);
        ldata->type = (light.type == fastgltf::LightType::Directional) ? LightingData::LightType::Directional
                      : (light.type == fastgltf::LightType::Point)     ? LightingData::LightType::Point
                                                                       : LightingData::LightType::Spot;
        ldata->name = light.name;
        file.lightingData[ldata->name] = ldata;
        lights.push_back(ldata);
    }

    GPUResourceAllocator &gpuResourceAllocator = GPUResourceAllocator::Instance();

    // Work out which images any texture actually resolves to before loading anything. With
    // MSFT_texture_dds the document lists both a PNG and a DDS per texture, and Bistro ships only
    // the DDS -- so loading blindly means 343 file opens that fail, log, and burn a slot on an
    // error image. Mirrors the DDS preference in bindTexture below; the two must agree.
    std::vector<bool> imageNeeded(gltf.images.size(), false);
    for (const fastgltf::Texture &tex : gltf.textures)
    {
        const fastgltf::Optional<std::size_t> &index = tex.ddsImageIndex.has_value() ? tex.ddsImageIndex : tex.imageIndex;
        if (index.has_value() && index.value() < imageNeeded.size())
        {
            imageNeeded[index.value()] = true;
        }
    }

    // images is indexed by glTF image index, so skipped entries still need a slot: keep it dense.
    for (size_t i = 0; i < gltf.images.size(); i++)
    {
        fastgltf::Image &image = gltf.images[i];

        if (!imageNeeded[i])
        {
            images.push_back(engine.GetErrorImage());
            continue;
        }

        std::optional<AllocatedImage> img = load_image(gltf, image, path.parent_path());

        if (img.has_value())
        {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        }
        else
        {
            // we failed to load, so lets give the slot a default white texture to not
            // completely break loading
            images.push_back(engine.GetErrorImage());
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
    }

    file.materialDataBuffer = gpuResourceAllocator.create_buffer(sizeof(MaterialSystem::MaterialConstants) * gltf.materials.size(),
                                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    MaterialSystem::MaterialConstants *sceneMaterialConstants = (MaterialSystem::MaterialConstants *)file.materialDataBuffer.info.pMappedData;

    for (fastgltf::Material &mat : gltf.materials)
    {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        MaterialSystem::MaterialConstants constants{}; // value-init: extra[] is read by the shaders now
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
        constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;

        // extra[0].x = alpha cutoff (0 disables the test). MaterialConstants already pads out 14
        // spare vec4s, so this costs no layout change.
        constants.extra[0].x = (mat.alphaMode == fastgltf::AlphaMode::Mask) ? float(mat.alphaCutoff) : 0.0f;

        // write material parameters to buffer
        sceneMaterialConstants[data_index] = constants;

        // MASK stays in the opaque bucket: it wants a discard, not blending.
        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend)
        {
            passType = MaterialPass::Transparent;
        }

        MaterialSystem::MaterialResources materialResources;
        // default the material textures
        materialResources.colorImage = engine.GetDefaultImage();
        materialResources.colorSampler = engine.GetDefaultSampler();
        materialResources.metalRoughImage = engine.GetDefaultImage();
        materialResources.metalRoughSampler = engine.GetDefaultSampler();
        materialResources.normalImage = engine.GetFlatNormalImage();
        materialResources.normalSampler = engine.GetDefaultSampler();
        materialResources.emissiveImage = engine.GetDefaultImage(); // white: emissive = factor when untextured
        materialResources.emissiveSampler = engine.GetDefaultSampler();

        // set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(MaterialSystem::MaterialConstants);
        materialResources.colorFactors = constants.colorFactors;
        // A texture entry may omit its sampler, in which case glTF says use defaults. Calling
        // .value() on that empty optional would be undefined behaviour.
        auto bindTexture = [&](size_t textureIndex, AllocatedImage &outImage, VkSampler &outSampler)
        {
            const fastgltf::Texture &tex = gltf.textures[textureIndex];

            // MSFT_texture_dds textures carry both: source is a PNG fallback, the extension names
            // the DDS that is actually on disk. Prefer the DDS, but fall back so plain glTFs work.
            const fastgltf::Optional<std::size_t> &index = tex.ddsImageIndex.has_value() ? tex.ddsImageIndex : tex.imageIndex;
            if (!index.has_value())
            {
                return;
            }
            outImage = images[index.value()];
            outSampler = tex.samplerIndex.has_value() ? file.samplers[tex.samplerIndex.value()] : engine.GetDefaultSampler();
        };

        // grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value())
        {
            bindTexture(mat.pbrData.baseColorTexture.value().textureIndex, materialResources.colorImage, materialResources.colorSampler);
        }
        if (mat.pbrData.metallicRoughnessTexture.has_value())
        {
            bindTexture(mat.pbrData.metallicRoughnessTexture.value().textureIndex, materialResources.metalRoughImage,
                        materialResources.metalRoughSampler);
        }
        if (mat.normalTexture.has_value())
        {
            bindTexture(mat.normalTexture.value().textureIndex, materialResources.normalImage, materialResources.normalSampler);
        }

        // extra[1].xyz = emissive colour. Bistro puts the multiplier straight in the factor (up to
        // 100) rather than using KHR_materials_emissive_strength, so do not clamp it to [0,1].
        glm::vec3 emissive(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
        emissive *= float(mat.emissiveStrength);

        if (mat.emissiveTexture.has_value())
        {
            bindTexture(mat.emissiveTexture.value().textureIndex, materialResources.emissiveImage, materialResources.emissiveSampler);

            // A few materials ship an emissive texture with a zero factor, which by spec multiplies
            // to nothing. That is clearly not the intent, so treat the factor as white.
            if (emissive == glm::vec3(0.0f))
            {
                emissive = glm::vec3(1.0f);
            }
        }

        constants.extra[1] = glm::vec4(emissive, 0.0f);
        sceneMaterialConstants[data_index] = constants;

        // build material
        newMat->data = engine.GetMaterialSystem().write_material(device, passType, materialResources, file.descriptorPool);

        data_index++;
    }

    // use the same vectors for all meshes so that the memory doesnt reallocate as
    // often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor &indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor, [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
            }

            // load vertex positions
            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                                                              [&](glm::vec3 v, size_t index)
                                                              {
                                                                  Vertex newvtx;
                                                                  newvtx.position = v;
                                                                  newvtx.normal = {1, 0, 0};
                                                                  newvtx.color = glm::vec4{1.f};
                                                                  newvtx.uv_x = 0;
                                                                  newvtx.uv_y = 0;
                                                                  vertices[initial_vtx + index] = newvtx;
                                                              });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).accessorIndex],
                                                              [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).accessorIndex],
                                                              [&](glm::vec2 v, size_t index)
                                                              {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                auto &accessor = gltf.accessors[(*colors).accessorIndex];
                if (accessor.type == fastgltf::AccessorType::Vec4)
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).accessorIndex],
                                                                  [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
                }
                else if (accessor.type == fastgltf::AccessorType::Vec3)
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*colors).accessorIndex], [&](glm::vec3 v, size_t index)
                                                                  { vertices[initial_vtx + index].color = glm::vec4(v, 1.0f); });
                }
            }

            if (p.materialIndex.has_value())
            {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                newSurface.material = materials[0];
            }

            // loop the vertices of this surface, find min/max bounds
            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (int i = initial_vtx; i < vertices.size(); i++)
            {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }
            // calculate origin and extents from the min/max, use extent lenght for radius
            newSurface.bounds.origin = (maxpos + minpos) / 2.f;
            newSurface.bounds.extents = (maxpos - minpos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = gpuResourceAllocator.uploadMesh(indices, vertices);
    }

    // load all nodes and their meshes
    for (fastgltf::Node &node : gltf.nodes)
    {
        std::shared_ptr<sgraph::Node> newNode;

        // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode
        // class
        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<sgraph::MeshNode>();
            static_cast<sgraph::MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else if (node.lightIndex.has_value())
        {
            // lighting mesh node.
            newNode = std::make_shared<sgraph::LightNode>();
            static_cast<sgraph::LightNode *>(newNode.get())->lightingData = lights[node.lightIndex.value()];
        }
        else
        {
            newNode = std::make_shared<sgraph::Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{[&](fastgltf::math::fmat4x4 matrix) { memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix)); },
                                     [&](fastgltf::TRS transform)
                                     {
                                         glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                                         glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                                         glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                                         glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                         glm::mat4 rm = glm::toMat4(rot);
                                         glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                         newNode->localTransform = tm * rm * sm;
                                     }},
                   node.transform);
    }

    // run loop again to setup transform hierarchy
    for (int i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node &node = gltf.nodes[i];
        std::shared_ptr<sgraph::Node> &sceneNode = nodes[i];

        for (auto &c : node.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes, with no parents
    for (auto &node : nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{1.f});
        }
    }

    return scene;
}

void sgraph::Scene::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    // create renderables from the scenenodes
    for (auto &n : topNodes)
    {
        n->Draw(topMatrix, ctx);
    }
}

void sgraph::Scene::clearAll()
{
    VulkanEngine &engine = VulkanEngine::Instance();
    VkDevice dv = engine.GetVkDevice();

    GPUResourceAllocator &gpuResourceAllocator = GPUResourceAllocator::Instance();

    descriptorPool.destroy_pools(dv);
    gpuResourceAllocator.destroy_buffer(materialDataBuffer);

    for (auto &[k, v] : meshes)
    {

        gpuResourceAllocator.destroy_buffer(v->meshBuffers.indexBuffer);
        gpuResourceAllocator.destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto &[k, v] : images)
    {

        if (v.image == engine.GetErrorImage().image)
        {
            // dont destroy the default images
            continue;
        }
        gpuResourceAllocator.destroy_image(v);
    }

    for (auto &sampler : samplers)
    {
        vkDestroySampler(dv, sampler, nullptr);
    }
}

VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter)
    {
    // nearest samplers
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

    // linear samplers
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

std::optional<AllocatedImage> load_image(fastgltf::Asset &asset, fastgltf::Image &image, const std::filesystem::path &baseDir)
{
    AllocatedImage newImage{};

    int width, height, nrChannels;

    GPUResourceAllocator &gpuResourceAllocator = GPUResourceAllocator::Instance();

    std::visit(
        fastgltf::visitor{
            [](auto &arg) {},
            [&](fastgltf::sources::URI &filePath)
            {
                assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                assert(filePath.uri.isLocalPath());   // We're only capable of loading
                                                      // local files.

                const std::string uriPath(filePath.uri.path().begin(),
                                          filePath.uri.path().end()); // Thanks C++.

                // glTF URIs are relative to the document, not to the working directory. Tolerate a
                // leading slash, and fall back to the literal URI in case it is already absolute.
                std::string relative = uriPath;
                while (!relative.empty() && (relative.front() == '/' || relative.front() == '\\'))
                {
                    relative.erase(relative.begin());
                }

                std::filesystem::path fsPath = baseDir / relative;
                if (!std::filesystem::exists(fsPath) && std::filesystem::exists(std::filesystem::path(uriPath)))
                {
                    fsPath = uriPath;
                }

                if (!std::filesystem::exists(fsPath))
                {
                    fmt::println("texture missing: {}", fsPath.string());
                    return;
                }

                // BC7 is block compressed, so stbi cannot touch it
                if (fsPath.extension() == ".dds" || fsPath.extension() == ".DDS")
                {
                    if (auto dds = load_dds(fsPath))
                    {
                        newImage = *dds;
                        return;
                    }
                    fmt::println("load_dds failed: {}", fsPath.string());
                    return;
                }

                const std::string path = fsPath.string();
                unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if (data)
                {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = gpuResourceAllocator.create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::Vector &vector)
            {
                unsigned char *data = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(vector.bytes.data()),
                                                            static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                if (data)
                {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = gpuResourceAllocator.create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::BufferView &view)
            {
                auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                auto &buffer = asset.buffers[bufferView.bufferIndex];

                std::visit(
                    fastgltf::visitor{
                        // We only care about VectorWithMime here, because we
                        // specify LoadExternalBuffers, meaning all buffers
                        // are already loaded into a vector.
                        [](auto &arg) {},
                        [&](fastgltf::sources::Vector &vector)
                        {
                            unsigned char *data =
                                stbi_load_from_memory(reinterpret_cast<const unsigned char *>(vector.bytes.data()) + bufferView.byteOffset,
                                                      static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
                            if (data)
                            {
                                VkExtent3D imagesize;
                                imagesize.width = width;
                                imagesize.height = height;
                                imagesize.depth = 1;

                                newImage =
                                    gpuResourceAllocator.create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);

                                stbi_image_free(data);
                            }
                        },
                        [&](fastgltf::sources::Array &array) // Added this case for newer fastgltf!
                        {
                            unsigned char *data =
                                stbi_load_from_memory(reinterpret_cast<const unsigned char *>(array.bytes.data()) + bufferView.byteOffset,
                                                      static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
                            if (data)
                            {
                                VkExtent3D imagesize;
                                imagesize.width = width;
                                imagesize.height = height;
                                imagesize.depth = 1;

                                newImage =
                                    gpuResourceAllocator.create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);

                                stbi_image_free(data);
                            }
                        },
                    },
                    buffer.data);
            },
        },
        image.data);

    // if any of the attempts to load the data failed, we havent written the image
    // so handle is null
    if (newImage.image == VK_NULL_HANDLE)
    {
        return {};
    }
    else
    {
        return newImage;
    }
}

std::optional<std::shared_ptr<AllocatedImage>> loadImage(std::string fileName)
{
    int height, width, channels;
    const auto &data = stbi_load(fileName.c_str(), &width, &height, &channels, 4);

    if (data)
    {
        VkExtent3D imageSize;
        imageSize.width = width;
        imageSize.height = height;
        imageSize.depth = 1;

        AllocatedImage newImage{};

        newImage = GPUResourceAllocator::Instance().create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);
        stbi_image_free(data);
        return std::make_shared<AllocatedImage>(newImage);
    }
    return {};
}