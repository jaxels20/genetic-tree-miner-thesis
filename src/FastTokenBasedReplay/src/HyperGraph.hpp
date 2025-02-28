#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include "Marking.hpp"

// Define MarkingHasher outside the HyperGraph class
struct MarkingHasher {
    size_t operator()(const Marking& m) const {
        size_t hash = 0;
        for (const auto& [place, tokens] : m.places) {
            hash ^= std::hash<std::string>{}(place) ^ std::hash<uint32_t>{}(tokens);
        }
        return hash;
    }
};


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
    
        void setMarking(const std::unordered_map<std::string, uint32_t>& marking) {
            for (const auto& [place, tokens] : marking) {
                if (hasNode(place)) {
                    setTokens(place, tokens);
                }
            }
        }
    
        void resetMarking() {
            for (auto& [name, node] : nodes) {
                node.tokens = 0;
            }
        }
    
        std::pair<bool, std::vector<std::string>> canReachTargetMarking(
            const Marking& start,
            const Marking& target) {
    
        using State = std::pair<Marking, std::vector<std::string>>;
        std::queue<State> queue;
        std::unordered_set<Marking, MarkingHasher> visited;
    
        queue.push({start, {}});
        visited.insert(start);
    
        while (!queue.empty()) {
            auto [current, path] = queue.front();
            queue.pop();
    
            // Check if the current marking satisfies the target condition
            bool targetReached = true;
            for (const auto& [place, tokens] : target.places) {
                // If the place is missing, treat it as zero tokens.
                uint32_t currentTokens = 0;
                if (current.places.find(place) != current.places.end()) {
                    currentTokens = current.places.at(place);
                }
                if (currentTokens < tokens) {
                    targetReached = false;
                    break;
                }
            }
            if (targetReached) {
                return {true, path};
            }
    
            // Try firing enabled edges
            for (const auto& [edgeName, edge] : edges) {
                bool enabled = true;
                for (const auto& src : edge.sources) {
                    // Check if each source exists and has at least one token.
                    if (current.places.find(src) == current.places.end() || current.places.at(src) == 0) {
                        enabled = false;
                        break;
                    }
                }
    
                if (enabled) {
                    Marking nextMarking = current;
    
                    // Decrement tokens from source places (they must exist, or the edge wouldn’t be enabled)
                    for (const auto& src : edge.sources) {
                        nextMarking.add_place(src, -1);
                    }
    
                    // Increment tokens to target places.
                    // This will add the target to the marking if it does not exist.
                    for (const auto& tgt : edge.targets) {
                        nextMarking.add_place(tgt, 1);
                    }
    
                    if (visited.find(nextMarking) == visited.end()) {
                        auto newPath = path;
                        newPath.push_back(edgeName);
                        queue.push({nextMarking, newPath});
                        visited.insert(nextMarking);
                    }
                }
            }
        }
        return {false, {}}; // No path found
    }

        

    private:
        std::unordered_map<std::string, Node> nodes;
        std::unordered_map<std::string, Edge> edges;
        
        struct HashMapHasher {
            size_t operator()(const std::unordered_map<std::string, uint32_t>& map) const {
                size_t hash = 0;
                for (const auto& [key, value] : map) {
                    hash ^= std::hash<std::string>{}(key) ^ std::hash<uint32_t>{}(value);
                }
                return hash;
            }
        };
    };
    