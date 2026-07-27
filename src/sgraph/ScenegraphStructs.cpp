#include "ScenegraphStructs.h"
#include "vk_engine.h"
#include "vk_loader.h"
#include "vk_types.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

void sgraph::MeshNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto &s : mesh->surfaces)
    {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.modelMatrix = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if (s.material->data.passType == MaterialPass::Transparent)
        {
            ctx.TransparentSurfaces.push_back(def);
        }
        else
        {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    // recurse down
    Node::Draw(topMatrix, ctx);
}

bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj)
{
    std::array<glm::vec3, 8> corners{
        glm::vec3{1, 1, 1},  glm::vec3{1, 1, -1},  glm::vec3{1, -1, 1},  glm::vec3{1, -1, -1},
        glm::vec3{-1, 1, 1}, glm::vec3{-1, 1, -1}, glm::vec3{-1, -1, 1}, glm::vec3{-1, -1, -1},
    };

    glm::mat4 matrix = viewproj * obj.modelMatrix;

    glm::vec3 min = {1.5, 1.5, 1.5};
    glm::vec3 max = {-1.5, -1.5, -1.5};

    for (int c = 0; c < 8; c++)
    {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3{v.x, v.y, v.z}, min);
        max = glm::max(glm::vec3{v.x, v.y, v.z}, max);
    }

    // check the clip space box is within the view
    if (min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void sgraph::LightNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    // each light node has 1 light.
    GPULightingData lData;
    lData.color = lightingData->color;
    lData.intensity = lightingData->intensity;
    lData.transform = topMatrix * this->worldTransform;
    lData.range = lightingData->range;
    ctx.lights.push_back(lData);
    Node::Draw(topMatrix, ctx);
}

// ---- Authored scenegraph node types --------------------------------------

void sgraph::TransformNode::applyTranslate(const glm::vec3 &t)
{
    localTransform = localTransform * glm::translate(glm::mat4(1.f), t);
}

void sgraph::TransformNode::applyRotate(float degrees, const glm::vec3 &axis)
{
    localTransform = localTransform * glm::rotate(glm::mat4(1.f), glm::radians(degrees), glm::normalize(axis));
}

void sgraph::TransformNode::applyScale(const glm::vec3 &s)
{
    localTransform = localTransform * glm::scale(glm::mat4(1.f), s);
}

void sgraph::GLTFLeafNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    // The referenced glTF is pure geometry: draw it at this leaf's accumulated
    // world transform (worldTransform is set by refreshTransform, or overwritten
    // by the physics sync for bodies driven by the simulation).
    if (geometry)
    {
        geometry->Draw(topMatrix * worldTransform, ctx);
    }
    // A leaf may still parent children in the authored graph.
    Node::Draw(topMatrix, ctx);
}