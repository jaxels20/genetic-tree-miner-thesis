#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

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
};

class Transition {
public:
    std::string name;

    Transition(std::string name = "") : name(name) {}

    std::string repr() const {
        return "Transition(" + name + ")";
    }

    bool operator==(const Transition& other) const {
        return name == other.name;
    }

    bool operator<(const Transition& other) const {
        return name < other.name;
    }

    bool operator>(const Transition& other) const {
        return name > other.name;
    }
};

class Arc {
public:
    std::string source;
    std::string target;
    int weight;

    Arc(std::string source, std::string target, int weight = 1)
        : source(source), target(target), weight(weight) {}

    std::string repr() const {
        return "Arc(" + source + " -> " + target + ", weight=" + std::to_string(weight) + ")";
    }
};

class PetriNet {
    public:
        std::vector<Place> places;
        std::vector<Transition> transitions;
        std::vector<Arc> arcs;
    
        void add_place(const Place& place) {
            places.push_back(place);
        }
    
        void add_transition(const Transition& transition) {
            transitions.push_back(transition);
        }
    
        void add_arc(const Arc& arc) {
            arcs.push_back(arc);
        }
    
        // Find a place by name
        Place* get_place(const std::string& name) {
            for (auto& place : places) {
                if (place.name == name) {
                    return &place;
                }
            }
            return nullptr;
        }
    
        // Find a transition by name
        Transition* get_transition(const std::string& name) {
            for (auto& transition : transitions) {
                if (transition.name == name) {
                    return &transition;
                }
            }
            return nullptr;
        }
    
        // Check if a transition can fire based on token availability
        bool can_fire(const Transition& transition) {
            for (const auto& arc : arcs) {
                if (arc.target == transition.name) {
                    Place* place = get_place(arc.source);
                    if (place && place->tokens < arc.weight) {
                        return false;  // Not enough tokens in place
                    }
                }
            }
            return true;
        }
    
        // Fire a transition, updating tokens in the places
        void fire_transition(const Transition& transition, int& consumed, int& produced) {
            for (auto& arc : arcs) {
                if (arc.target == transition.name) {
                    Place* place = get_place(arc.source);
                    if (place) {
                        place->remove_tokens(arc.weight);
                        consumed += arc.weight;
                    }
                }
                if (arc.source == transition.name) {
                    Place* place = get_place(arc.target);
                    if (place) {
                        place->add_tokens(arc.weight);
                        produced += arc.weight;
                    }
                }
            }
        }
    
        std::string repr() const {
            return "PetriNet(places=" + std::to_string(places.size()) +
                   ", transitions=" + std::to_string(transitions.size()) +
                   ", arcs=" + std::to_string(arcs.size()) + ")";
        }
    };