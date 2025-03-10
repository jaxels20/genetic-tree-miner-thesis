
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
    auto [reachable, sequence] = silent_graph.canReachTargetMarking(curr_marking, final_marking, 1);
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

Graph create_silent_graph(const PetriNet& net) {
    Graph silent_graph; // Create a graph object
    PetriNet net_copy = net;
    std::vector<Place> places = net_copy.get_all_places();
    std::vector<Transition> silent_transitions = net_copy.get_all_silent_transitions();

    // for each silent transition add an edge between the different permutations of the input and output sets
    for (const auto& transition : silent_transitions) {
        std::vector<Place> input_places = net_copy.get_preset(transition);
        std::vector<Place> output_places = net_copy.get_postset(transition);

        std::cout << std::endl;

        for (const auto& input_place : input_places) {
            for (const auto& output_place : output_places) {
                silent_graph.addEdge(input_place.name, output_place.name, 1, transition.name);
            }
        }
    }

    return silent_graph;
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
replay_trace_without_caching(const Trace& trace, PetriNet& net, HyperGraph silent_graph) {   
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

        if(net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            std::vector<Place> input_places = net.get_preset(*transition);
            Marking curr_marking = net.get_current_marking();

            // check if there is a path from the current marking to the marking enabling the transition
            
            auto [reachable, sequence] = silent_graph.canReachTargetMarking(curr_marking, net.get_marking_enabling_transition(*transition), 1);

            if (reachable) {
                net.fire_transition_sequence(sequence, &consumed, &produced);
            } else {
                // Create a token in the pre-set of the transition
                for (const auto& place : net.get_preset(*transition)) {
                    Place* p = net.get_place(place.name);
                    if (p) {
                        if (p->number_of_tokens() == 0) {
                            p->add_tokens(1);
                            missing += 1;
                        }
                    }
                }
            }

            // Fire the transition
            net.fire_transition(*transition, &consumed, &produced);
        }
    }

    consumed += net.final_marking.number_of_tokens();

    // Finalize the tokens in the Petri net
    finalize_tokens(net, silent_graph, missing, consumed, produced);

    // Count the remaining tokens in the Petri net
    int32_t remaining_tokens = net.number_of_tokens() - net.final_marking.number_of_tokens();

    remaining += remaining_tokens;

    // print the results
    //std::cout << "Missing: " << missing << std::endl;
    //std::cout << "Remaining: " << remaining << std::endl;
    //std::cout << "Produced: " << produced << std::endl;
    //std::cout << "Consumed: " << consumed << std::endl;

    return std::make_tuple(static_cast<double>(missing), static_cast<double>(remaining), static_cast<double>(produced), static_cast<double>(consumed));
}

std::tuple<double, double, double, double> 
replay_trace_with_prefix(const Trace& trace, PetriNet& net, HyperGraph silent_graph, PrefixTree& prefix_cache) {
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

        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            auto [reachable, sequence] = silent_graph.canReachTargetMarking(net.get_current_marking(), net.get_marking_enabling_transition(*transition), 1);
            if (reachable) {
                net.fire_transition_sequence(sequence, &consumed, &produced);
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

        std::vector<Event> prefix(trace.events.begin(), trace.events.begin() + i + 1);
        auto new_prefix_node = prefix_cache.get_or_create_node(prefix);
        new_prefix_node->replay_data = {missing, remaining, produced, consumed};
        new_prefix_node->marking = net.get_current_marking();
    }

    consumed += net.final_marking.number_of_tokens();
    finalize_tokens(net, silent_graph, missing, consumed, produced);

    remaining += net.number_of_tokens() - net.final_marking.number_of_tokens();

    return std::make_tuple(static_cast<double>(missing), static_cast<double>(remaining), static_cast<double>(produced), static_cast<double>(consumed));
}

std::tuple<double, double, double, double> 
replay_trace_with_suffix(const Trace& trace, PetriNet& net, HyperGraph silent_graph, std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher>& postfixCache) {
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
        
        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            auto [reachable, sequence] = silent_graph.canReachTargetMarking(net.get_current_marking(),
                                                net.get_marking_enabling_transition(*transition), 1);
            if (reachable) {
                net.fire_transition_sequence(sequence, &consumed, &produced);
            } else {
                for (const auto& place : net.get_preset(*transition)) {
                    Place* p = net.get_place(place.name);
                    if (p && p->number_of_tokens() == 0) {
                        p->add_tokens(1);
                        missing += 1;
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }
    }

    consumed += net.final_marking.number_of_tokens();
    finalize_tokens(net, silent_graph, missing, consumed, produced);

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
replay_trace_with_prefix_and_suffix(const Trace& trace, PetriNet& net, HyperGraph silent_graph, PrefixTree& prefix_cache, std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher>& suffix_cache) {
    int missing = 0, remaining = 0, consumed = 0, produced = 0;
    
    produced += net.initial_marking.number_of_tokens();
    initialize_tokens(net);

    // Use prefix cache to jump-start the replay if possible.
    std::vector<Event> matched_prefix;
    std::shared_ptr<PrefixNode> prefix_node = prefix_cache.get_longest_matching_prefix(trace.events, matched_prefix);
    size_t start_index = 0;
    if (!matched_prefix.empty()) {
        // Retrieve cached replay data and marking from the prefix node.
        auto [cached_missing, cached_remaining, cached_produced, cached_consumed] = prefix_node->replay_data;
        missing = cached_missing;
        remaining = cached_remaining;
        produced = cached_produced;
        consumed = cached_consumed;
        net.set_marking(prefix_node->marking);
        start_index = matched_prefix.size();
    }

    // Replay the remaining events, checking for a suffix match before processing each event.
    for (size_t i = start_index; i < trace.events.size(); ++i) {
        // Compute a key representing the remainder of the trace starting at index i.
        std::string postfix = computePostfix(trace, i);
        MarkingPostfixKey key{ net.get_current_marking(), postfix };

        // If a cached suffix sequence is available, fire it and update the metrics.
        auto it = suffix_cache.find(key);
        if (it != suffix_cache.end()) {
            const std::vector<std::string>& cachedSequence = it->second;
            net.fire_transition_sequence(cachedSequence, &consumed, &produced);
            break; // Assume the suffix completes the replay.
        }

        // Process the event normally.
        const auto& event = trace.events[i];
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);
        } else {
            // Check if a silent (invisible) transition sequence can reach the enabling marking.
            auto [reachable, sequence] = silent_graph.canReachTargetMarking(
                net.get_current_marking(),
                net.get_marking_enabling_transition(*transition),
                1
            );
            if (reachable) {
                net.fire_transition_sequence(sequence, &consumed, &produced);
            } else {
                // Add a token to an input place if necessary.
                for (const auto& place : net.get_preset(*transition)) {
                    Place* p = net.get_place(place.name);
                    if (p && p->number_of_tokens() == 0) {
                        p->add_tokens(1);
                        missing += 1;
                    }
                }
                net.fire_transition(*transition, &consumed, &produced);
            }
        }

        // Update prefix cache with the replay information for the current prefix.
        std::vector<Event> prefix(trace.events.begin(), trace.events.begin() + i + 1);
        auto new_prefix_node = prefix_cache.get_or_create_node(prefix);
        new_prefix_node->replay_data = {missing, remaining, produced, consumed};
        new_prefix_node->marking = net.get_current_marking();

        // (Optional) If you are tracking the sequence of transitions fired,
        // you could update the suffix cache here so that the same suffix can be reused later.
        // For example:
        // suffixCache[key] = sequence_fired_so_far;
    }

    consumed += net.final_marking.number_of_tokens();
    finalize_tokens(net, silent_graph, missing, consumed, produced);

    int32_t remaining_tokens = net.number_of_tokens() - net.final_marking.number_of_tokens();
    remaining += remaining_tokens;

    return std::make_tuple(
        static_cast<double>(missing),
        static_cast<double>(remaining),
        static_cast<double>(produced),
        static_cast<double>(consumed)
    );
}

double 
calculate_fitness(const EventLog& log, const PetriNet& net, bool prefix_caching, bool suffix_caching){
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    HyperGraph silent_graph = create_silent_hyper_graph(net);
    PrefixTree prefix_cache;
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher> postfixCache;
    // Map to store computed values for unique traces
    std::unordered_map<Trace, std::tuple<int, int, int, int>> trace_cache;

    // Iterate over the traces in the event log
    for (const auto& trace : log.traces) {
        if (trace_cache.find(trace) == trace_cache.end()) {
            // If this trace has not been processed, do token replay
            PetriNet net_copy = net;
            if (prefix_caching && suffix_caching) {
                trace_cache[trace] = replay_trace_with_prefix_and_suffix(trace, net_copy, silent_graph, prefix_cache, postfixCache);
            } else if (prefix_caching) {
                trace_cache[trace] = replay_trace_with_prefix(trace, net_copy, silent_graph, prefix_cache);
            } else if (suffix_caching) {
                trace_cache[trace] = replay_trace_with_suffix(trace, net_copy, silent_graph, postfixCache);
            } else {
                trace_cache[trace] = replay_trace_without_caching(trace, net_copy, silent_graph);
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









