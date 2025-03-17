
#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "Graph.hpp"
#include "HyperGraph.hpp"
#include "PrefixTree.hpp"
#include "SuffixTree.hpp"
#include <set>
#include "silent_transition_handling.cpp"
#include "ActivityCache.hpp"

bool stop_condition_final_marking(Marking& current_marking, Marking& final_marking) {
    // Check if the current marking is equal (or a subset of) the final marking
    for (const auto& [place, tokens] : final_marking.places) {
        if (current_marking.places.find(place) == current_marking.places.end()) {
            return false;
        }
        if (current_marking.places[place] < tokens) {
            return false;
        }
    }
    return true;
    
}

void initialize_tokens(PetriNet& net) {
    // produce tokens in the initial marking
    for (const auto& [place, tokens] : net.initial_marking.places) {
        Place* p = net.get_place(place);
        if (p) {
            p->add_tokens(tokens);
        }
    }
}

void finalize_tokens(PetriNet& net, HyperGraph& silent_graph, int& missing, int& consumed, int& produced) {
    // Check if there are tokens in the final marking
    // If not, try to use silent transitions before adding tokens manually

    Marking final_marking = net.final_marking;
    Marking curr_marking = net.get_current_marking();

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Check if the final marking is reachable from the current marking
    auto [reachable, sequence] = silent_graph.canReachTargetMarking(curr_marking, final_marking, 5);
    if (reachable) {
        net.fire_transition_sequence(sequence, &consumed, &produced);
    }

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Else create tokens in the places of the final markin

    for (const auto& [place, tokens] : net.final_marking.places) {
        Place* p = net.get_place(place);
        if (!p) continue;

        int32_t tokens_in_place = p->number_of_tokens();
        if (tokens_in_place < tokens) {
            p->add_tokens(tokens - tokens_in_place);
            missing += tokens - tokens_in_place;
        }
    }
}

void finalize_tokens_v2(PetriNet& net, std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences, int& missing, int& consumed, int& produced) {
    // Check if there are tokens in the final marking
    // If not, try to use silent transitions before adding tokens manually

    Marking final_marking = net.final_marking;
    Marking curr_marking = net.get_current_marking();

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Check if the final marking is reachable from the current marking
    // auto [reachable, sequence] = attempt_to_reach_final_marking_by_firing_silent_transitions(net, silent_firing_sequences, final_marking);
    // if (reachable) {
    //     net.fire_transition_sequence(sequence, &consumed, &produced);
    // }

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Else create tokens in the places of the final markin

    for (const auto& [place, tokens] : net.final_marking.places) {
        Place* p = net.get_place(place);
        if (!p) continue;

        int32_t tokens_in_place = p->number_of_tokens();
        if (tokens_in_place < tokens) {
            p->add_tokens(tokens - tokens_in_place);
            missing += tokens - tokens_in_place;
        }
    }
}

HyperGraph create_silent_hyper_graph(const PetriNet& net) {
    HyperGraph hypergraph;
    PetriNet net_copy = net;
    // Add places as nodes in the hypergraph
    for (const auto& place : net_copy.places) {
        hypergraph.addNode(place.name, place.tokens);
    }

    // Add silent transitions as hyperedges in the hypergraph
    for (const auto& transition : net_copy.transitions) {
        if (transition.is_silent()) {
            // Identify the input places (sources) and output places (targets) for the silent transition
            std::vector<std::string> sourcePlaces;
            std::vector<std::string> targetPlaces;

            // Gather input places (sources)
            for (const auto& arc : net_copy.arcs) {
                if (arc.target == transition.name) {
                    sourcePlaces.push_back(arc.source);
                }
            }

            // Gather output places (targets)
            for (const auto& arc : net_copy.arcs) {
                if (arc.source == transition.name) {
                    targetPlaces.push_back(arc.target);
                }
            }

            // Add the hyperedge to the hypergraph
            hypergraph.addEdge(transition.name, sourcePlaces, targetPlaces);
        }
    }

    return hypergraph;
}

std::string computePostfix(const Trace& trace, size_t currentIndex) {
    std::string result;
    for (size_t i = currentIndex; i < trace.events.size(); ++i) {
        result += trace.events[i].activity + ",";
    }
    return result;
}

std::tuple<double, double, double, double> 
replay_trace_without_caching(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache) {   
        int missing = 0;   // Count of missing tokens (tokens added to input places to enable transitions)
    int remaining = 0; // Count of remaining tokens in the Petri net at the end
    int consumed = 0;  // Count of tokens consumed from input places
    int produced = 0;  // Count of tokens produced in output places

    produced += net.initial_marking.number_of_tokens();

    // Initialize the tokens in the Petri net
    initialize_tokens(net);

    // Iterate over the events in the trace
    for (const auto& event : trace.events) {
        
        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                    activity_cache.store(current_marking, transition->name, sequence);
                } else {
                    for (const auto& place : net.get_preset(*transition)) {
                        Place* p = net.get_place(place.name);
                        if (p && p->number_of_tokens() == 0) {
                            p->add_tokens(1);
                            missing += 1;
                        }
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }
    }

    consumed += net.final_marking.number_of_tokens();

    // Finalize the tokens in the Petri net
    finalize_tokens_v2(net, silent_firing_sequences, missing, consumed, produced);

    // Count the remaining tokens in the Petri net
    int32_t remaining_tokens = net.number_of_tokens() - net.final_marking.number_of_tokens();

    remaining += remaining_tokens;

    return std::make_tuple(
        static_cast<double>(missing), 
        static_cast<double>(remaining), 
        static_cast<double>(produced), 
        static_cast<double>(consumed));
}

std::tuple<double, double, double, double> 
replay_trace_with_prefix(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache,
    PrefixTree& prefix_cache) {
    int missing = 0, remaining = 0, consumed = 0, produced = 0;
    
    produced += net.initial_marking.number_of_tokens();
    initialize_tokens(net);

    std::vector<Event> matched_prefix;
    std::shared_ptr<PrefixNode> prefix_node = prefix_cache.get_longest_matching_prefix(trace.events, matched_prefix);
    
    if (!matched_prefix.empty()) {
        auto [cached_missing, cached_remaining, cached_produced, cached_consumed] = prefix_node->replay_data;
        missing = cached_missing;
        remaining = cached_remaining;
        produced = cached_produced;
        consumed = cached_consumed;
        
        net.set_marking(prefix_node->marking);
    }

    for (size_t i = matched_prefix.size(); i < trace.events.size(); i++) {
        const auto& event = trace.events[i];

        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                    activity_cache.store(current_marking, transition->name, sequence);
                } else {
                    for (const auto& place : net.get_preset(*transition)) {
                        Place* p = net.get_place(place.name);
                        if (p && p->number_of_tokens() == 0) {
                            p->add_tokens(1);
                            missing += 1;
                        }
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }

        std::vector<Event> prefix(trace.events.begin(), trace.events.begin() + i + 1);
        auto new_prefix_node = prefix_cache.get_or_create_node(prefix);
        new_prefix_node->replay_data = {missing, remaining, produced, consumed};
        new_prefix_node->marking = net.get_current_marking();
    }

    consumed += net.final_marking.number_of_tokens();
    finalize_tokens_v2(net, silent_firing_sequences, missing, consumed, produced);

    remaining += net.number_of_tokens() - net.final_marking.number_of_tokens();

    return std::make_tuple(static_cast<double>(missing), static_cast<double>(remaining), static_cast<double>(produced), static_cast<double>(consumed));
}

std::tuple<double, double, double, double> 
replay_trace_with_suffix(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache, 
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher>& postfixCache) {
    int missing = 0, remaining = 0, consumed = 0, produced = 0;
    produced += net.initial_marking.number_of_tokens();

    initialize_tokens(net);

    // Iterate over events in the trace
    for (size_t i = 0; i < trace.events.size(); ++i) {
        // Before processing, check if a cached postfix is available
        std::string postfix = computePostfix(trace, i);
        MarkingPostfixKey key { net.get_current_marking(), postfix };
        auto it = postfixCache.find(key);
        if (it != postfixCache.end()) {
            // Cached sequence found – fire it and update metrics.
            const std::vector<std::string>& cachedSequence = it->second;
            net.fire_transition_sequence(cachedSequence, &consumed, &produced);
            // Assume replay finishes after applying the cached sequence.
            break;
        }

        // Process current event normally
        Transition* transition = net.get_transition(trace.events[i].activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + trace.events[i].activity);
        }
        
        if(net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                    activity_cache.store(current_marking, transition->name, sequence);
                } else {
                    for (const auto& place : net.get_preset(*transition)) {
                        Place* p = net.get_place(place.name);
                        if (p && p->number_of_tokens() == 0) {
                            p->add_tokens(1);
                            missing += 1;
                        }
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }
    }

    consumed += net.final_marking.number_of_tokens();
    finalize_tokens_v2(net, silent_firing_sequences, missing, consumed, produced);

    int32_t remaining_tokens = net.number_of_tokens() - net.final_marking.number_of_tokens();
    remaining += remaining_tokens;

    // Optionally, after finishing a replay you might store the (marking, postfix) key along with
    // the final sequence of transitions that were fired (if it is beneficial to cache).
    // For example, if you replayed the entire trace (or a suffix of it) and you want to cache it:
    Marking currentMarking = net.get_current_marking();
    std::string emptyPostfix = "";  // At the end the remaining postfix is empty.
    // Here, one might store the sequence of transitions that were fired to finish the case.
    // This example assumes you have tracked that sequence.
    // postfixCache[{currentMarking, emptyPostfix}] = finalTransitionSequence;

    return std::make_tuple(static_cast<double>(missing),
                           static_cast<double>(remaining),
                           static_cast<double>(produced),
                           static_cast<double>(consumed));
}

std::tuple<double, double, double, double>
replay_trace_with_prefix_and_suffix(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, 
    std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences, 
    ActivityCache& activity_cache) {

    int missing = 0;   // Count of missing tokens (tokens added to input places to enable transitions)
    int remaining = 0; // Count of remaining tokens in the Petri net at the end
    int consumed = 0;  // Count of tokens consumed from input places
    int produced = 0;  // Count of tokens produced in output places

    produced += net.initial_marking.number_of_tokens();

    // Initialize the tokens in the Petri net
    initialize_tokens(net);

    // Iterate over the events in the trace
    for (const auto& event : trace.events) {
        
        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                    activity_cache.store(current_marking, transition->name, sequence);
                } else {
                    for (const auto& place : net.get_preset(*transition)) {
                        Place* p = net.get_place(place.name);
                        if (p && p->number_of_tokens() == 0) {
                            p->add_tokens(1);
                            missing += 1;
                        }
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }
    }

    consumed += net.final_marking.number_of_tokens();

    // Finalize the tokens in the Petri net
    finalize_tokens_v2(net, silent_firing_sequences, missing, consumed, produced);

    // Count the remaining tokens in the Petri net
    int32_t remaining_tokens = net.number_of_tokens() - net.final_marking.number_of_tokens();

    remaining += remaining_tokens;

    return std::make_tuple(
        static_cast<double>(missing), 
        static_cast<double>(remaining), 
        static_cast<double>(produced), 
        static_cast<double>(consumed));
}

double 
calculate_fitness(const EventLog& log, const PetriNet& net, bool prefix_caching, bool suffix_caching){
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    HyperGraph silent_hyper_graph = create_silent_hyper_graph(net);
    PrefixTree prefix_cache;
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher> postfixCache;
    // Map to store computed values for unique traces
    std::unordered_map<Trace, std::tuple<int, int, int, int>> trace_cache;

    Graph silent_graph = create_silent_graph(net);

    PetriNet net_copy = net;
    // A map to store the firing sequences for every place to every other place using silent transitions
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences;
    silent_firing_sequences = get_places_shortest_path_by_hidden(net_copy, 5);

    // Activity cache to store the precomputed values
    ActivityCache activity_cache;

    // Iterate over the traces in the event log
    for (const auto& trace : log.traces) {
        if (trace_cache.find(trace) == trace_cache.end()) {
            // If this trace has not been processed, do token replay
            PetriNet net_copy = net;
            if (prefix_caching && suffix_caching) {
                trace_cache[trace] = replay_trace_with_prefix_and_suffix(trace, net_copy, silent_firing_sequences, activity_cache);
            } else if (prefix_caching) {
                trace_cache[trace] = replay_trace_with_prefix(trace, net_copy, silent_firing_sequences, activity_cache,  prefix_cache);
            } else if (suffix_caching) {
                trace_cache[trace] = replay_trace_with_suffix(trace, net_copy, silent_firing_sequences, activity_cache, postfixCache);
            } else {
                trace_cache[trace] = replay_trace_without_caching(trace, net_copy, silent_firing_sequences, activity_cache);
            }
        }

        // Retrieve precomputed values
        auto [missing, remaining, produced, consumed] = trace_cache[trace];

        // Update the total counts
        total_missing += missing;
        total_remaining += remaining;
        total_produced += produced;
        total_consumed += consumed;
    }

    double fitness = 0.5 * (1 - (static_cast<double>(total_missing) / total_consumed)) + 0.5 * (1 - (static_cast<double>(total_remaining) / total_produced));

    return fitness;
}









