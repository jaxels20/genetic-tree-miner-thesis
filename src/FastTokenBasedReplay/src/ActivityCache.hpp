#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "Marking.hpp"


class ActivityCache {
    public:
        std::unordered_map<Marking, std::unordered_map<std::string, std::vector<std::string>>, MarkingHasher> cache;
    
        void store(const Marking& marking, const std::string& transition, const std::vector<std::string>& silent_sequence) {
            cache[marking][transition] = silent_sequence;
        }
    
        std::vector<std::string>* retrieve(const Marking& marking, const std::string& transition) {
            if (cache.find(marking) != cache.end() && cache[marking].find(transition) != cache[marking].end()) {
                return &cache[marking][transition];
            }
            return {}; // Return empty vector if no cache entry exists
        }
    };