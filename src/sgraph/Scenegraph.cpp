#include "Scenegraph.h"
#include "ScenegraphStructs.h"
#include <memory>
#include <optional>
#include <string>

using namespace sgraph;

void Scenegraph::makeScenegraph(std::unordered_map<std::string, std::shared_ptr<INode>> &&scenegraphNodes)
{
    this->nodes = std::move(scenegraphNodes);
}

std::shared_ptr<INode> Scenegraph::getRoot()
{
    return root;
}

std::optional<std::shared_ptr<INode>> Scenegraph::getNode(const std::string &name)
{
    auto found = nodes.find(name);
    if (found != nodes.end())
    {
        return found->second;
    }
    return std::nullopt;
}

void Scenegraph::setRoot(std::shared_ptr<INode> root)
{
    this->root = root;
}

Scenegraph::~Scenegraph()
{
    nodes.clear();
}

void Scenegraph::cleanup()
{
    nodes.clear();
}