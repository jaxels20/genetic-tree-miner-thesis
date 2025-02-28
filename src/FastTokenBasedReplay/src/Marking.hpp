#pragma once

#include <string>
#include <unordered_map>
#include <initializer_list>

class Marking {
    public:
        std::unordered_map<std::string, uint32_t> places;
        
        Marking() = default;

        Marking(std::initializer_list<std::pair<std::string, uint32_t>> init) {
            for (const auto& [place, tokens] : init) {
                places[place] = tokens;
            }
        }
    
        void add_place(const std::string& place, uint32_t tokens) {
            places[place] += tokens;
        }
    
        uint32_t number_of_tokens() const {
            uint32_t tokens = 0;
            for (const auto& [_, count] : places) {
                tokens += count;
            }
            return tokens;
        }
    
        bool contains(const Marking& target) const {
            for (const auto& [place, tokens] : target.places) {
                if (places.at(place) < tokens) return false;
            }
            return true;
        }
    };