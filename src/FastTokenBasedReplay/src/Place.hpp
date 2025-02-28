#pragma once

#include <string>
#include <stdexcept>



class Place {
    public:
        std::string name;
        int tokens;
    
        Place(std::string name, int tokens = 0) : name(name), tokens(tokens) {}
    
        void add_tokens(int count = 1) {
            tokens += count;
        }
    
        void remove_tokens(int count = 1) {
            if (tokens - count < 0) {
                throw std::runtime_error("Cannot remove " + std::to_string(count) +
                                         " tokens from place '" + name + "' (tokens = " +
                                         std::to_string(tokens) + ")");
            }
            tokens -= count;
        }
    
        std::string repr() const {
            return "Place(" + name + ", tokens=" + std::to_string(tokens) + ")";
        }
    
        int32_t number_of_tokens() const {
            return tokens;
        }
    
    };
    