#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <memory>
#include "Marking.hpp"
#include "Eventlog.hpp"

class PrefixNode {
    public:
        std::unordered_map<std::string, std::shared_ptr<PrefixNode>> children;
        std::tuple<int, int, int, int> replay_data; // (missing, remaining, produced, consumed)
        Marking marking;  // Store the marking at this prefix
        
        PrefixNode() : replay_data({0, 0, 0, 0}) {}
    };
    
    class PrefixTree {
    public:
        std::shared_ptr<PrefixNode> root;
    
        PrefixTree() { root = std::make_shared<PrefixNode>(); }
    
        std::shared_ptr<PrefixNode> get_or_create_node(const std::vector<Event>& events) {
            auto node = root;
            for (const auto& event : events) {
                if (node->children.find(event.activity) == node->children.end()) {
                    node->children[event.activity] = std::make_shared<PrefixNode>();
                }
                node = node->children[event.activity];
            }
            return node;
        }
    
        std::shared_ptr<PrefixNode> get_longest_matching_prefix(const std::vector<Event>& events, std::vector<Event>& matched_prefix) {
            auto node = root;
            matched_prefix.clear();
            
            for (const auto& event : events) {
                if (node->children.find(event.activity) == node->children.end()) {
                    break;
                }
                node = node->children[event.activity];
                matched_prefix.push_back(event);
            }
            return node;
        }
    };
    