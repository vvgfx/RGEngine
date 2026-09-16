#pragma once
#include "IScenegraph.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace sgraph
{
    class Scenegraph : public IScenegraph
    {
      public:
        virtual void makeScenegraph(std::unordered_map<std::string, std::shared_ptr<INode>> &&scenegraphNodes) override;

        // virtual void addNode(const std::string name, std::shared_ptr<INode> node) override;

        virtual std::shared_ptr<INode> getRoot() override;

        // virtual std::unordered_map<std::string, std::shared_ptr<INode>> getNodes() override;

        virtual std::optional<std::shared_ptr<INode>> getNode(const std::string &name) override;

        virtual void setRoot(std::shared_ptr<INode> root) override;

        ~Scenegraph();

        virtual void cleanup() override;

      private:
        std::unordered_map<std::string, std::shared_ptr<INode>> nodes;
        std::shared_ptr<INode> root;
    };

} // namespace sgraph