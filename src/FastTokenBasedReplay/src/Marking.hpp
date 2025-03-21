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
                auto it = places.find(place);
                if (it == places.end() || it->second < tokens) {
                    return false;
                }
            }
            return true;
        }
    
        // Implement the equality operator
        bool operator==(const Marking& other) const {
            return places == other.places;
        }

        int get_tokens(const std::string& place) const {
            if (places.find(place) != places.end()) {
                return places.at(place);
            }
            return 0;
        }

        std::string to_string() const {
            std::string str = "";
            for (const auto& [place, tokens] : places) {
                str += place + ": " + std::to_string(tokens) + ", ";
            }
            return str;
        }
    };


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

struct MarkingPairHasher {
    size_t operator()(const std::pair<Marking, Marking>& p) const {
        MarkingHasher markingHasher;
        size_t hash1 = markingHasher(p.first);
        size_t hash2 = markingHasher(p.second);
        return hash1 ^ (hash2 << 1); // Combine hashes uniquely
    }
};
