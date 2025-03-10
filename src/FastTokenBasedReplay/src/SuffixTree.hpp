#pragma once
#include "Marking.hpp"
#include <string>

struct MarkingPostfixKey {
    Marking marking;
    std::string postfix; // e.g. concatenated activities of the remaining trace

    bool operator==(const MarkingPostfixKey& other) const {
        return marking == other.marking && postfix == other.postfix;
    }
};

// A hasher for the key
struct MarkingPostfixKeyHasher {
    size_t operator()(const MarkingPostfixKey& key) const {
        MarkingHasher mh;
        size_t h1 = mh(key.marking);
        size_t h2 = std::hash<std::string>()(key.postfix);
        return h1 ^ (h2 << 1);
    }
};

