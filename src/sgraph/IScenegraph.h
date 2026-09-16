#pragma once
#include "ScenegraphStructs.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace sgraph
{
    class IScenegraph
    {
      public:
        virtual ~IScenegraph()
        {
        }

        /**
         * @brief Sets this input INode as the root.
         *
         * @param root root INode
         */
        virtual void makeScenegraph(std::unordered_map<std::string, std::shared_ptr<INode>> &&) = 0;

        // /**
        //  * @brief Add this INode to the scenegraph. This does not set the parent hierarchy, which must be manually
        //  done.
        //  *
        //  * @param name name of the INode.
        //  * @param node the INode.
        //  */
        // virtual void addNode(const std::string name, std::shared_ptr<INode> node) = 0;

        /**
         * @brief Get the Root INode
         *
         * @return std::shared_ptr<INode> Root INode
         */
        virtual std::shared_ptr<INode> getRoot() = 0;

        /**
         * @brief Set the Root Node
         *
         */
        virtual void setRoot(std::shared_ptr<INode> root) = 0;

        // I don't need this for now, will re-add it later if required.
        // virtual std::unordered_map<std::string, std::shared_ptr<INode>> getNodes() = 0;

        /**
         * @brief Get the INode of the corresponding name
         *
         * @param name name of the INode
         * @return std::shared_ptr<INode> the INode.
         */
        virtual std::optional<std::shared_ptr<INode>> getNode(const std::string &name) = 0;

        /**
         * @brief force cleanup on scenegraph nodes.
         *
         */
        virtual void cleanup() = 0;
    };
} // namespace sgraph