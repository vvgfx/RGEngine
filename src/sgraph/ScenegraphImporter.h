#pragma once

#include "Scenegraph.h"
#include "ScenegraphStructs.h"
#include "vk_loader.h"
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sgraph
{

    // Parses the authored scenegraph command stream. 'rigidbody' and 'magnet' become neutral
    // specs, so nothing here depends on Box3D.
    // Grammar:
    //   gltf <name> <path>
    //   group|node|transform <var>                       equivalent spellings; all make a plain Node
    //   translate <var> <x> <y> <z>
    //   rotate <var> <degrees> <ax> <ay> <az>
    //   scale <var> <sx> <sy> <sz>
    //   mesh <var> <gltfName>
    //   add-child <child> <parent>
    //   root <var>
    //   rigidbody <var> <static|dynamic|kinematic> <box|sphere|capsule|hull> [density]
    class ScenegraphImporter
    {
      public:
        ScenegraphImporter(GLTFCreatorData &data) : creatorData(data)
        {
        }

        std::shared_ptr<Scenegraph> parse(std::istream &input)
        {
            std::string cleaned = stripComments(input);
            std::istringstream lines(cleaned);
            std::string line;
            auto scenegraph = std::make_shared<Scenegraph>();

            while (std::getline(lines, line))
            {
                std::vector<std::string> tok = tokenize(line);
                if (tok.empty())
                {
                    continue;
                }
                const std::string &cmd = tok[0];

                if (cmd == "gltf")
                    parseGLTF(tok);
                else if (cmd == "group" || cmd == "node" || cmd == "transform")
                    parseNode(tok);
                else if (cmd == "translate")
                    parseTranslate(tok);
                else if (cmd == "rotate")
                    parseRotate(tok);
                else if (cmd == "scale")
                    parseScale(tok);
                else if (cmd == "mesh")
                    parseMesh(tok);
                else if (cmd == "add-child")
                    parseAddChild(tok);
                else if (cmd == "root")
                    parseSetRoot(tok);
                else if (cmd == "rigidbody")
                    parseRigidBody(tok);
                else
                    std::cout << "ScenegraphImporter: invalid command '" << cmd << "'\n";
            }

            scenegraph->makeScenegraph(nodes);
            scenegraph->setRoot(root);
            return scenegraph;
        }

        // in declaration order; the physics layer consumes these
        const std::vector<RigidBodySpec> &getPhysicsSpecs() const
        {
            return physicsSpecs;
        }

      private:
        void parseGLTF(const std::vector<std::string> &tok)
        {
            if (tok.size() < 3)
            {
                std::cout << "gltf: expected <name> <path>\n";
                return;
            }
            std::string_view fileView = tok[2];
            auto gltf = loadGltf(creatorData, fileView);
            if (!gltf.has_value())
            {
                std::cout << "gltf: unable to load '" << tok[2] << "'\n";
                return;
            }
            gltf->get()->name = tok[1];
            geometries[tok[1]] = gltf.value();
        }

        // three spellings for the same thing, for authoring readability
        void parseNode(const std::vector<std::string> &tok)
        {
            if (tok.size() < 2)
                return;
            nodes[tok[1]] = std::make_shared<Node>();
        }

        void parseTranslate(const std::vector<std::string> &tok)
        {
            if (tok.size() < 5)
            {
                std::cout << "translate: expected <var> <x> <y> <z>\n";
                return;
            }
            if (auto t = asNode(tok[1]))
                t->applyTranslate({std::stof(tok[2]), std::stof(tok[3]), std::stof(tok[4])});
        }

        void parseRotate(const std::vector<std::string> &tok)
        {
            if (tok.size() < 6)
            {
                std::cout << "rotate: expected <var> <degrees> <ax> <ay> <az>\n";
                return;
            }
            if (auto t = asNode(tok[1]))
                t->applyRotate(std::stof(tok[2]), {std::stof(tok[3]), std::stof(tok[4]), std::stof(tok[5])});
        }

        void parseScale(const std::vector<std::string> &tok)
        {
            if (tok.size() < 5)
            {
                std::cout << "scale: expected <var> <sx> <sy> <sz>\n";
                return;
            }
            if (auto t = asNode(tok[1]))
                t->applyScale({std::stof(tok[2]), std::stof(tok[3]), std::stof(tok[4])});
        }

        // one Scene, one place in the graph: binding the same geometry twice is an error, not sharing
        void parseMesh(const std::vector<std::string> &tok)
        {
            if (tok.size() < 3)
            {
                std::cout << "mesh: expected <var> <gltfName>\n";
                return;
            }
            auto it = geometries.find(tok[2]);
            if (it == geometries.end())
            {
                std::cout << "mesh: unknown gltf geometry '" << tok[2] << "'\n";
                return;
            }
            if (!boundGeometries.insert(tok[2]).second)
            {
                std::cout << "mesh: geometry '" << tok[2] << "' is already bound to a node; register it under a "
                             "second 'gltf' name to place it twice\n";
                return;
            }
            nodes[tok[1]] = it->second;
        }

        void parseAddChild(const std::vector<std::string> &tok)
        {
            if (tok.size() < 3)
                return;
            auto childNode = std::dynamic_pointer_cast<Node>(getNode(tok[1]));
            auto parentNode = std::dynamic_pointer_cast<Node>(getNode(tok[2]));
            if (parentNode && childNode)
            {
                parentNode->children.push_back(childNode);
                childNode->parent = parentNode;
            }
            else
            {
                std::cout << "add-child: could not link '" << tok[1] << "' -> '" << tok[2] << "'\n";
            }
        }

        void parseSetRoot(const std::vector<std::string> &tok)
        {
            if (tok.size() < 2)
                return;
            if (!nodes.contains(tok[1]))
            {
                std::cout << "root: missing node '" << tok[1] << "'\n";
                return;
            }
            root = nodes[tok[1]];
        }

        void parseRigidBody(const std::vector<std::string> &tok)
        {
            if (tok.size() < 4)
            {
                std::cout << "rigidbody: expected <var> <type> <shape> [density]\n";
                return;
            }
            RigidBodySpec spec;
            spec.nodeName = tok[1];

            if (tok[2] == "static")
                spec.body = RigidBodySpec::Body::Static;
            else if (tok[2] == "dynamic")
                spec.body = RigidBodySpec::Body::Dynamic;
            else if (tok[2] == "kinematic")
                spec.body = RigidBodySpec::Body::Kinematic;
            else
                std::cout << "rigidbody: unknown body type '" << tok[2] << "'\n";

            if (tok[3] == "box")
                spec.shape = RigidBodySpec::Shape::Box;
            else if (tok[3] == "sphere")
                spec.shape = RigidBodySpec::Shape::Sphere;
            else if (tok[3] == "capsule")
                spec.shape = RigidBodySpec::Shape::Capsule;
            else if (tok[3] == "hull")
                spec.shape = RigidBodySpec::Shape::Hull;
            else
                std::cout << "rigidbody: unknown shape '" << tok[3] << "'\n";

            if (tok.size() >= 5)
                spec.density = std::stof(tok[4]);

            physicsSpecs.push_back(spec);
        }

        // ---- helpers ---------------------------------------------------------
        std::shared_ptr<Node> asNode(const std::string &name)
        {
            auto t = std::dynamic_pointer_cast<Node>(getNode(name));
            if (!t)
                std::cout << "'" << name << "' is not a node\n";
            return t;
        }

        std::shared_ptr<INode> getNode(const std::string &name)
        {
            auto it = nodes.find(name);
            return it == nodes.end() ? nullptr : it->second;
        }

        static std::vector<std::string> tokenize(const std::string &line)
        {
            std::istringstream ls(line);
            std::vector<std::string> tok;
            std::string t;
            while (ls >> t)
                tok.push_back(t);
            return tok;
        }

        std::string stripComments(std::istream &input)
        {
            std::string line;
            std::stringstream clean;
            while (std::getline(input, line))
            {
                std::size_t i = 0;
                while (i < line.length() && line[i] != '#')
                {
                    clean << line[i];
                    i++;
                }
                clean << "\n";
            }
            return clean.str();
        }

        GLTFCreatorData &creatorData;
        std::unordered_map<std::string, std::shared_ptr<INode>> nodes;
        std::unordered_map<std::string, std::shared_ptr<Scene>> geometries;
        std::unordered_set<std::string> boundGeometries; // geometry names already placed in the graph by 'mesh'
        std::vector<RigidBodySpec> physicsSpecs;
        std::shared_ptr<INode> root;
    };
} // namespace sgraph
