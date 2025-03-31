#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "Marking.hpp"
#include "Place.hpp"
#include <set>


class Transition {
public:
    std::string name;

    Transition(std::string name = "") : name(name) {}

    std::string repr() const {
        return "Transition(" + name + ")";
    }

    // If the name starts with "tau", it is a silent transition
    bool is_silent() const {
        return name.rfind("tau", 0) == 0; // Checks if "tau" is at the start
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

        Marking initial_marking;
        Marking final_marking;
    
        void add_place(const Place& place) {
            places.push_back(place);
        }
    
        void add_transition(const Transition& transition) {
            transitions.push_back(transition);
        }
    
        void add_arc(const Arc& arc) {
            arcs.push_back(arc);
        }
        
        void set_initial_marking(const Marking& marking) {
            initial_marking = marking;
        }
        
        void set_final_marking(const Marking& marking) {
            final_marking = marking;
        }

        // Find a place by name
        Place* get_place(const std::string& name){
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
        void fire_transition(const Transition& transition, int* consumed, int* produced) {
            for (auto& arc : arcs) {
                if (arc.target == transition.name) {
                    Place* place = get_place(arc.source);
                    if (place) {
                        place->remove_tokens(arc.weight);
                        if (consumed) {
                            *consumed += arc.weight;  // Dereferencing the pointer
                        }
                    }
                }
                if (arc.source == transition.name) {
                    Place* place = get_place(arc.target);
                    if (place) {
                        place->add_tokens(arc.weight);
                        if (produced) {
                            *produced += arc.weight;  // Dereferencing the pointer
                        }
                    }
                }
            }
        }

        std::string repr() const {
            return "PetriNet(places=" + std::to_string(places.size()) +
                   ", transitions=" + std::to_string(transitions.size()) +
                   ", arcs=" + std::to_string(arcs.size()) + ")";
        }

        std::vector<Place> get_postset(const Transition& transition) {
            std::vector<Place> postset;
            for (const auto& arc : arcs) {
                if (arc.source == transition.name) {
                    Place* place = get_place(arc.target);
                    if (place) {
                        postset.push_back(*place);
                    }
                }
            }
            return postset;
        }

        std::vector<Place> get_preset(const Transition& transition) {
            std::vector<Place> preset;
            for (const auto& arc : arcs) {
                if (arc.target == transition.name) {
                    Place* place = get_place(arc.source);
                    if (place) {
                        preset.push_back(*place);
                    }
                }
            }
            return preset;
        }
        
        uint32_t number_of_tokens() const {
            uint32_t tokens = 0;
            for (const auto& place : places) {
                tokens += place.tokens;
            }
            return tokens;
        }
    
        std::vector<Transition> get_all_silent_transitions() {
            std::vector<Transition> silent_transitions;
            for (const auto& transition : transitions) {
                if (transition.is_silent()) {
                    silent_transitions.push_back(transition);
                }
            }
            return silent_transitions;
        }

        std::vector<Place> get_all_places() {
            return places;
        }

        Marking get_current_marking() {
            // Loop through all places and add them to the marking
            Marking marking;
            for (const auto& place : places) {
                if (place.tokens > 0) {
                    marking.add_place(place.name, place.tokens);
                }
            }
            return marking;
        }

        std::vector<std::string> fire_transition_sequence(const std::vector<std::string>& transition_names, int* consumed, int* produced) {
            std::vector<std::string> fired_transitions;
            for (const auto& transition_name : transition_names) {
                Transition* transition = get_transition(transition_name);
                if (transition) {
                    if (can_fire(*transition)) {
                        fire_transition(*transition, consumed, produced);
                        fired_transitions.push_back(transition_name);
                    }
                    else {
                        throw std::runtime_error("Transition cannot fire: " + transition_name);
                    }
                }
                else {
                    throw std::runtime_error("Transition not found: " + transition_name);
                }
            }
            return fired_transitions;
        }
        
        // Fire a sequence of transitions, stopping if a transition cannot fire returns the fired transitions
        std::vector<std::string> partially_fire_transition_sequence(const std::vector<std::string>& transition_names, int* consumed, int* produced) {
            std::vector<std::string> fired_transitions;
            for (const auto& transition_name : transition_names) {
                Transition* transition = get_transition(transition_name);
                if (transition) {
                    if (can_fire(*transition)) {
                        fire_transition(*transition, consumed, produced);
                        fired_transitions.push_back(transition_name);
                    }
                    else {
                        return fired_transitions;
                    }
                }
                else {
                    return fired_transitions;
                }
            }
            return fired_transitions;
        }


        bool can_fire_transition_sequence(const std::vector<std::string>& transition_names) {

            // Make a copy of self 
            PetriNet net_copy = *this;

            // Try to fire each transition in the sequence
            for (const auto& transition_name : transition_names) {
                Transition* transition = net_copy.get_transition(transition_name);
                if (transition) {
                    if (net_copy.can_fire(*transition)) {
                        net_copy.fire_transition(*transition, nullptr, nullptr);
                    }
                    else {
                        return false;
                    }
                }
                else {
                    return false;
                }
            }
            
            return true;
        }

        void set_marking(const Marking& marking) {
            for (auto& place : places) {
                place.tokens = marking.get_tokens(place.name);
            }
        }
    
        Marking get_marking_enabling_transition(const Transition& transition) {
            Marking marking;
            std::vector<Place> preset = get_preset(transition);
            for (const auto& place : preset) {
                marking.add_place(place.name, 1);
            }
            return marking;
        }
    
        std::vector<std::string> get_enabled_transitions(bool include_silent = false) {
            // Return the names of all enabled transitions (excluding silent transitions)
            std::vector<std::string> enabled_transitions;
            for (const auto& transition : transitions) {
                if (transition.is_silent() && !include_silent) {
                    continue;  // Skip silent transitions
                }
                if (can_fire(transition)) {
                    enabled_transitions.push_back(transition.name);
                }
            }
            return enabled_transitions;
        }
        
        Marking fire_transition_without_changing_marking(const Transition& transition, const Marking& marking) {
            // save the current marking
            Marking old_marking = get_current_marking();
            // set the marking to the new one
            set_marking(marking);
            // fire the transition
            fire_transition(transition, nullptr, nullptr);
            // get the new marking
            Marking new_marking = get_current_marking();
            // set the marking back to the original one
            set_marking(old_marking);
            return new_marking;
        }

        bool can_fire_transition_from_marking(Marking marking, const Transition& transition) {
            // copy the current marking
            Marking marking_copy = get_current_marking();

            // set the marking to the new one
            set_marking(marking);
            bool is_transition_fireable = can_fire(transition);
            // set the marking back to the original one
            set_marking(marking_copy);
            return is_transition_fireable;
        }

        std::vector<std::string> get_enabled_transitions_in_marking(const Marking& marking, bool include_silent = false) {
            // old marking 
            Marking old_marking = get_current_marking();
            // set the marking to the new one
            set_marking(marking);
            // get the enabled transitions
            auto enabled_transitions = get_enabled_transitions(include_silent);
            // set the marking back to the original one
            set_marking(old_marking);
            // return the enabled transitions
            return enabled_transitions;
        }

        std::set<std::string> get_visible_transitions_eventually_enabled() {
            std::set<std::string> visible_transitions;
            std::set<std::string> visited;
            
            // Start with the initially enabled transitions
            std::vector<std::string> all_enabled_transitions = get_enabled_transitions(true);
            std::unordered_map<std::string, Marking> transition_markings;
            
            for (const auto& t : all_enabled_transitions) {
                transition_markings[t] = get_current_marking();
            }
        
            size_t i = 0;
            while (i < all_enabled_transitions.size()) {
                std::string t = all_enabled_transitions[i];
                Marking marking_copy = transition_markings[t];
        
                // Check if we've already visited this transition-marking combination
                std::string key = t + marking_copy.to_string();
                if (visited.find(key) == visited.end()) {
                    Transition* transition = get_transition(t);
                    if (!transition) {
                        ++i;
                        continue;
                    }
        
                    if (!transition->is_silent()) {
                        visible_transitions.insert(t);
                    } else {
                        if (can_fire_transition_from_marking(marking_copy, *transition)) {
                            // Fire the transition without changing the marking
                            Marking new_marking = fire_transition_without_changing_marking(*transition, marking_copy);
                            std::vector<std::string> new_enabled_transitions = get_enabled_transitions_in_marking(new_marking, true);
                            
                            for (const auto& t2 : new_enabled_transitions) {
                                if (transition_markings.find(t2) == transition_markings.end()) {
                                    all_enabled_transitions.push_back(t2);
                                    transition_markings[t2] = new_marking;
                                }
                            }
                        }
                    }
                    visited.insert(key);
                }
                ++i;
            }
        
            return visible_transitions;
        }
        
    };