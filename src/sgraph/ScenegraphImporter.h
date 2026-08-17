#pragma once

#include "IScenegraph.h"
#include "Scenegraph.h"
#include "ScenegraphStructs.h"
#include "glm/gtx/transform.hpp"
#include "vk_loader.h"
#include <iostream>
#include <istream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

using namespace std;
namespace sgraph
{

    class ScenegraphImporter
    {

      public:
        shared_ptr<IScenegraph> parse(istream &input)
        {
            string command;
            string inputWithOutCommentsString = stripComments(input);
            istringstream inputWithOutComments(inputWithOutCommentsString);
            // Scenegraph *scenegraph = new Scenegraph();
            shared_ptr<Scenegraph> scenegraph = make_shared<Scenegraph>();
            while (inputWithOutComments >> command)
            {
                cout << "Read " << command << endl;
                if (command == "gltf")
                {
                    parseGLTF(inputWithOutComments);
                }
                else if (command == "add-child")
                {
                    parseAddChild(inputWithOutComments);
                }
                else if (command == "node")
                {
                    parseNode(inputWithOutComments);
                }
                else if (command == "root")
                {
                    parseSetRoot(inputWithOutComments);
                }
                else
                {
                    cout << "invalid command" << endl;
                }
                // could add meshnode here, but not doing so because all are imported via gltf only right now anyways.
            }

            // now build the scenegraph
            scenegraph->makeScenegraph(nodes);
            scenegraph->setRoot(root);
            return scenegraph;
        }

        virtual void parseGLTF(istream &input)
        {
            string name, filePath;
            input >> name >> filePath;
            string_view fileView = filePath;
            auto gltfNode = loadGltf(fileView);
            if (!gltfNode.has_value())
            {
                cout << "Unable to load gltf node at : " << filePath << endl;
                return;
            }
            gltfNode->get()->name = name;
            nodes[name] = gltfNode.value();
        }

        virtual void parseAddChild(istream &input)
        {
            // TODO: Come back to this and add proper structure to add children to all nodes.
            // right now this works for all "Nodes", which are normal "groupNodes" and "lightNodes" since lightnodes
            // current inherit from Node
            string childname, parentname;

            input >> childname >> parentname;
            auto parentNode = std::dynamic_pointer_cast<Node>(nodes[parentname]);

            auto childNode = std::dynamic_pointer_cast<Node>(nodes[childname]);

            if (parentNode && childNode)
            {
                parentNode->children.push_back(childNode);
                childNode->parent = parentNode;
            }
        }

        virtual void parseNode(istream &input)
        {
            string name;
            input >> name;

            shared_ptr<Node> node = std::make_shared<sgraph::Node>();

            nodes[name] = node;
        }

        virtual void parseSetRoot(istream &input)
        {
            string rootName;

            input >> rootName;

            if (!nodes.contains(rootName))
            {
                cout << "Root node is missing." << endl;
                return;
            }

            root = nodes[rootName];
        }

        string stripComments(istream &input)
        {
            string line;
            stringstream clean;
            while (getline(input, line))
            {
                int i = 0;
                while ((i < line.length()) && (line[i] != '#'))
                {
                    clean << line[i];
                    i++;
                }
                clean << endl;
            }
            return clean.str();
        }

      private:
        unordered_map<string, std::shared_ptr<INode>> nodes;
        std::shared_ptr<INode> root;
    };
} // namespace sgraph