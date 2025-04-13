#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "PetriNet.hpp"
#include "Graph.hpp"
#include <set>
#include "Marking.hpp"


struct CompareVectorLength {
    bool operator()(const std::vector<std::string>& a, const std::vector<std::string>& b) const {
        return a.size() < b.size(); // Sort by length
    }
};

Graph create_silent_graph(const PetriNet& net) {
    Graph silent_graph; // Create a graph object
    PetriNet net_copy = net;
    std::vector<Place> places = net_copy.get_all_places();
    std::vector<Transition> silent_transitions = net_copy.get_all_silent_transitions();

    // for each silent transition add an edge between the different permutations of the input and output sets
    for (const auto& transition : silent_transitions) {
        std::vector<Place> input_places = net_copy.get_preset(transition);
        std::vector<Place> output_places = net_copy.get_postset(transition);

        for (const auto& input_place : input_places) {
            for (const auto& output_place : output_places) {
                silent_graph.addEdge(input_place.name, output_place.name, 1, transition.name);
            }
        }
    }

    return silent_graph;
}

std::set<std::string> compute_delta_set(const Marking& current_marking, const Marking& target_marking) {
    std::set<std::string> delta_set;
    for (const auto& [place, tokens] : target_marking.places) {
        if (current_marking.places.find(place) == current_marking.places.end()) {
            delta_set.insert(place);
        } else {
            if (current_marking.places.at(place) < tokens) {
                delta_set.insert(place);
            }
        }
    }
    return delta_set;
}

std::set<std::string> compute_lambda_set(const Marking& current_marking, const Marking& target_marking) {
    std::set<std::string> lambda_set;
    for (const auto& [place, tokens] : current_marking.places) {
        auto target_it = target_marking.places.find(place);
        // Include if the place is not in target or has excess tokens
        if (tokens > 0 && (target_it == target_marking.places.end() || tokens > target_it->second)) {
            lambda_set.insert(place);
        }
    }
    return lambda_set;
}

std::set<std::vector<std::string>, CompareVectorLength> 
get_possible_firing_sequences(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>& firing_sequences, 
    const std::set<std::string>& delta_set, 
    const std::set<std::string>& lambda_set) 
{
    std::set<std::vector<std::string>, CompareVectorLength> possible_firing_sequences;
    
    for (const auto& lambda : lambda_set) {
        for (const auto& delta : delta_set) {
            // Now looking for paths from lambda to delta
            if (firing_sequences.find(lambda) != firing_sequences.end() && 
                firing_sequences.at(lambda).find(delta) != firing_sequences.at(lambda).end()) 
            {
                possible_firing_sequences.insert(firing_sequences.at(lambda).at(delta));
            }
        }
    }
    
    return possible_firing_sequences;
}

std::tuple<bool, std::vector<std::string>> 
attempt_to_make_transition_enabled_by_firing_silent_transitions(PetriNet& net, Transition* transition, std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> firing_sequences) {
    PetriNet net_copy = net;
    std::vector<std::string> final_firing_sequence;
    Marking current_marking = net.get_current_marking();
    Marking target_marking = net.get_marking_enabling_transition(*transition);
    std::set<std::string> delta_set = compute_delta_set(current_marking, target_marking);
    std::set<std::string> lambda_set = compute_lambda_set(current_marking, target_marking);
    std::set<std::vector<std::string>, CompareVectorLength> possible_firing_sequences = get_possible_firing_sequences(firing_sequences, delta_set, lambda_set);
    
    size_t max_iterations = 10;
    size_t iterations = 0;

    while (!delta_set.empty()){
        iterations++;
        if (iterations >= max_iterations) {
            break;
        }
        for (const auto& sequence : possible_firing_sequences) {
            // Partially fire the sequence 
            std::vector<std::string> fired_transitions = net_copy.partially_fire_transition_sequence(sequence, nullptr, nullptr);
            
            // if no transitions were fired, continue to the next sequence
            if (fired_transitions.empty()) {
                continue;
            }
            
            // append the sequence to the final firing sequence
            final_firing_sequence.insert(final_firing_sequence.end(), fired_transitions.begin(), fired_transitions.end());

            // check the T transition is enabled
            if (net_copy.can_fire(*transition)) {
                return std::make_tuple(true, final_firing_sequence);
            }
            
            // update the current marking
            current_marking = net_copy.get_current_marking();
            delta_set = compute_delta_set(current_marking, target_marking);
            lambda_set = compute_lambda_set(current_marking, target_marking);
            possible_firing_sequences = get_possible_firing_sequences(firing_sequences, delta_set, lambda_set);
            break; // break the for loop to start the while loop again
        }
    }
    // No sequence found to enable the transition
    return std::make_tuple(false, std::vector<std::string>());

};

std::tuple<bool, std::vector<std::string>>
attempt_to_reach_final_marking_by_firing_silent_transitions(PetriNet& net, std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> firing_sequences, Marking final_marking) {
    PetriNet net_copy = net;
    bool reachable_by_silent_transitions = false;
    std::vector<std::string> final_firing_sequence;

    Marking current_marking = net.get_current_marking();
    Marking target_marking = final_marking;

    std::set<std::string> delta_set = compute_delta_set(current_marking, final_marking);
    std::set<std::string> lambda_set = compute_lambda_set(current_marking, final_marking);
    std::set<std::vector<std::string>, CompareVectorLength> possible_firing_sequences = get_possible_firing_sequences(firing_sequences, delta_set, lambda_set);

    size_t max_iterations = 5;
    size_t iterations = 0;

     while (!delta_set.empty()){
        iterations++;
        if (iterations >= max_iterations) {
            break;
        }
        for (const auto& sequence : possible_firing_sequences) {
            // Partially fire the sequence 
            std::vector<std::string> fired_transitions = net_copy.partially_fire_transition_sequence(sequence, nullptr, nullptr);
            
            // if no transitions were fired, continue to the next sequence
            if (fired_transitions.empty()) {
                continue;
            }
            // append the sequence to the final firing sequence
            final_firing_sequence.insert(final_firing_sequence.end(), fired_transitions.begin(), fired_transitions.end());

            // check the we have reached the final marking
            if (net_copy.get_current_marking().contains(final_marking)) {
                return std::make_tuple(true, final_firing_sequence);
            }
            
            // update the current marking
            current_marking = net_copy.get_current_marking();
            delta_set = compute_delta_set(current_marking, target_marking);
            lambda_set = compute_lambda_set(current_marking, target_marking);
            possible_firing_sequences = get_possible_firing_sequences(firing_sequences, delta_set, lambda_set);
            break; // break the for loop to start the while loop again
        }
    }
    // No sequence found to get to the final marking
    return std::make_tuple(false, std::vector<std::string>());

};

void get_places_shortest_path(
    PetriNet& net,
    const std::string& place_to_populate,
    const std::string& current_place,
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>& places_shortest_path,
    std::vector<std::string> actual_list,
    int rec_depth,
    int max_rec_depth
) {
    if (rec_depth > max_rec_depth) {
        return;
    }
    if (places_shortest_path.find(place_to_populate) == places_shortest_path.end()) {
        places_shortest_path[place_to_populate] = {};
    }
    
    for (const auto& arc : net.arcs) {
        if (arc.source == current_place) {
            Transition* transition = net.get_transition(arc.target);
            if (transition && transition->is_silent()) {
                for (const auto& out_arc : net.arcs) {
                    if (out_arc.source == transition->name) {
                        std::string next_place = out_arc.target;
                        if (places_shortest_path[place_to_populate].find(next_place) == places_shortest_path[place_to_populate].end() ||
                            actual_list.size() + 1 < places_shortest_path[place_to_populate][next_place].size()) {
                            
                            std::vector<std::string> new_actual_list = actual_list;
                            new_actual_list.push_back(transition->name);
                            places_shortest_path[place_to_populate][next_place] = new_actual_list;
                            
                            get_places_shortest_path(net, place_to_populate, next_place, places_shortest_path, new_actual_list, rec_depth + 1, max_rec_depth);
                        }
                    }
                }
            }
        }
    }
}

std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>
 get_places_shortest_path_by_hidden(PetriNet& net, int max_rec_depth) {
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> places_shortest_path;
    for (const auto& place : net.places) {
        get_places_shortest_path(net, place.name, place.name, places_shortest_path, {}, 0, max_rec_depth);
    }
    return places_shortest_path;
}

