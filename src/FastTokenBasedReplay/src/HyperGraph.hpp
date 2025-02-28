#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>



class HyperGraph {
    public:
        struct Node {
            std::string name;
            int tokens;
        };
    
        struct Edge {
            std::string name;
            std::unordered_set<std::string> sources;
            std::unordered_set<std::string> targets;
        };
    
        void addNode(const std::string& nodeName, int tokens = 0) {
            nodes[nodeName] = {nodeName, tokens};
        }
    
        void addEdge(const std::string& edgeName, const std::vector<std::string>& sourceNodes, const std::vector<std::string>& targetNodes) {
            edges[edgeName] = {edgeName, 
                               std::unordered_set<std::string>(sourceNodes.begin(), sourceNodes.end()),
                               std::unordered_set<std::string>(targetNodes.begin(), targetNodes.end())};
        }
    
        bool hasNode(const std::string& nodeName) const {
            return nodes.find(nodeName) != nodes.end();
        }
    
        bool hasEdge(const std::string& edgeName) const {
            return edges.find(edgeName) != edges.end();
        }
    
        int getTokens(const std::string& nodeName) const {
            return nodes.at(nodeName).tokens;
        }
    
        void setTokens(const std::string& nodeName, int tokens) {
            nodes.at(nodeName).tokens = tokens;
        }
    
        std::unordered_set<std::string> getEdgeSources(const std::string& edgeName) const {
            return edges.at(edgeName).sources;
        }
    
        std::unordered_set<std::string> getEdgeTargets(const std::string& edgeName) const {
            return edges.at(edgeName).targets;
        }
    
    private:
        std::unordered_map<std::string, Node> nodes;
        std::unordered_map<std::string, Edge> edges;
    };