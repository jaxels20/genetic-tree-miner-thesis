
#pragma once

#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "SuffixTree.hpp"
#include "silent_transition_handling.cpp"
#include "ActivityCache.hpp"
#include <sstream>



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

void finalize_tokens(PetriNet& net, std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences, int& missing, int& consumed, int& produced) {
    // Check if there are tokens in the final marking
    // If not, try to use silent transitions before adding tokens manually

    Marking final_marking = net.final_marking;
    Marking curr_marking = net.get_current_marking();

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Try to reach the final marking by firing silent transitions
    auto [reachable, firing_sequence] = attempt_to_reach_final_marking_by_firing_silent_transitions(net, silent_firing_sequences, final_marking);

    if (reachable) {
        net.fire_transition_sequence(firing_sequence, &consumed, &produced);
    }

    // check if the final mrking is contained in the current marking
    if (stop_condition_final_marking(curr_marking, final_marking)) {
        return;
    }

    // Else create tokens in the places of the final marking
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

std::string computePostfix(const Trace& trace, size_t currentIndex) {
    std::string result;
    for (size_t i = currentIndex; i < trace.events.size(); ++i) {
        result += trace.events[i].activity + ",";
    }
    return result;
}

std::string computePrefix(const Trace& trace, size_t currentIndex) {
    std::string result;
    for (size_t i = 0; i <= currentIndex; ++i) {
        result += trace.events[i].activity + ",";
    }
    return result;
}

std::optional<std::tuple<size_t, std::tuple<int, int, int, int>, Marking>> 
get_longest_prefix(
    const std::unordered_map<std::string, std::tuple<std::tuple<int, int, int, int>, Marking>>& new_prefix_cache,
    const Trace& trace
) {
    std::ostringstream prefix_key;
    bool found_last = false;  // Tracks whether the previous step had a match
    std::optional<std::tuple<size_t, std::tuple<int, int, int, int>, Marking>> best_match = std::nullopt;

    for (size_t i = 0; i < trace.events.size(); ++i) {
        if (i > 0) prefix_key << ",";  // Add a comma separator after the first element
        prefix_key << trace.events[i].activity;

        auto it = new_prefix_cache.find(prefix_key.str());
        if (it != new_prefix_cache.end()) {
            best_match = std::make_tuple(
                i + 1,                     // Length of the longest prefix
                std::get<0>(it->second),   // First element (std::tuple<int, int, int, int>)
                std::get<1>(it->second)    // Second element (Marking)
            );
            found_last = true;
        } else if (!found_last) {
            break; // No match at this step, and adding more won’t help
        } else {
            found_last = false; // Allow one more iteration in case there's a late match
        }
    }

    return best_match;
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
    finalize_tokens(net, silent_firing_sequences, missing, consumed, produced);

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
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache,
    std::unordered_map<std::string, std::tuple<std::tuple<int, int, int, int>, Marking>>& new_prefix_cache) {

    int missing = 0;   // Count of missing tokens (tokens added to input places to enable transitions)
    int remaining = 0; // Count of remaining tokens in the Petri net at the end
    int consumed = 0;  // Count of tokens consumed from input places
    int produced = 0;  // Count of tokens produced in output places
    
    std::string curr_prefix = "";
    size_t start_index = 0;

    produced += net.initial_marking.number_of_tokens();

    // Initialize the tokens in the Petri net
    initialize_tokens(net);

    // Check if a cached prefix is available
    auto prefix_match = get_longest_prefix(new_prefix_cache, trace);
    if (prefix_match) {
        auto [prefix_length, replay_data, marking] = *prefix_match;
        missing   = std::get<0>(replay_data);
        remaining = std::get<1>(replay_data);
        produced  = std::get<2>(replay_data);
        consumed  = std::get<3>(replay_data);
        
        net.set_marking(marking);

        std::ostringstream oss;
        for (size_t i = 0; i < prefix_length; ++i) {
            if (i > 0) oss << ",";
            oss << trace.events[i].activity;
        }
        curr_prefix = oss.str();

        start_index = prefix_length; // Start from the next event
    }
    
    // Iterate over the events in the trace
    for (size_t i = start_index; i < trace.events.size(); ++i) {
        Event event = trace.events[i];

        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }
        // if the net cannot fire try to enable the transition by firing silent transitions
        if (!net.can_fire(*transition)) {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                    activity_cache.store(current_marking, transition->name, sequence);
                }
            }
        }

        // if it still cant be fired, add tokens to the input places
        if (!net.can_fire(*transition)) {
            for (const auto& place : net.get_preset(*transition)) {
                Place* p = net.get_place(place.name);
                if (p && p->number_of_tokens() == 0) {
                    p->add_tokens(1);
                    missing += 1;
                }
            }
        }

        // now fire the transitio
        if (net.can_fire(*transition)) {
            net.fire_transition(*transition, &consumed, &produced);

            // Add the current event to the prefix
            if (!curr_prefix.empty()) {
                curr_prefix += ",";
            }
            curr_prefix += event.activity;
            // Store the current prefix and its replay data
            new_prefix_cache[curr_prefix] = std::make_tuple(
                std::make_tuple(missing, remaining, produced, consumed),
                net.get_current_marking()
            );

        }else {
            throw std::runtime_error("Transition cannot be fired: " + event.activity);
        }
        
    }

    consumed += net.final_marking.number_of_tokens();

    // Finalize the tokens in the Petri net
    finalize_tokens(net, silent_firing_sequences, missing, consumed, produced);

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
replay_trace_with_suffix(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache, 
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher>& suffix_cache) {
    int missing = 0, remaining = 0, consumed = 0, produced = 0;

    // The local cache for suffixes (which in the end is appended to the global cache)
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher> local_suffix_cache;
    
    produced += net.initial_marking.number_of_tokens();    
    initialize_tokens(net);

    // Iterate over the events in the trace
    for (size_t i = 0; i < trace.events.size(); ++i) {
        Event event = trace.events[i];

        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        // Compute the postfix and chech if it is already in the cache
        std::string postfix = computePostfix(trace, i);
        MarkingPostfixKey key = {net.get_current_marking(), postfix};
        auto it = suffix_cache.find(key);
        if (it != suffix_cache.end()) {
            // If the postfix is found in the cache, fire the corresponding sequence
            const std::vector<std::string>& firing_sequence = it->second;
            net.fire_transition_sequence(firing_sequence, &consumed, &produced);
            break;
        }

        // if the net cannot fire try to enable the transition by firing silent transitions
        if (!net.can_fire(*transition)) {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, &consumed, &produced);
                
                // add the fired sequence to the local cache
                for (auto& [key, firing_sequence] : local_suffix_cache) {
                    firing_sequence.insert(firing_sequence.end(), cached_sequence.begin(), cached_sequence.end());
                }
                
                
            
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);
                if (reachable) {
                    net.fire_transition_sequence(sequence, &consumed, &produced);
                                    // add the fired sequence to the local cache
                    for (auto& [key, firing_sequence] : local_suffix_cache) {
                        firing_sequence.insert(firing_sequence.end(), cached_sequence.begin(), cached_sequence.end());
                    }
                    activity_cache.store(current_marking, transition->name, sequence);
                }
            }
        }

        // if it still cant be fired, add tokens to the input places
        if (!net.can_fire(*transition)) {
            for (const auto& place : net.get_preset(*transition)) {
                Place* p = net.get_place(place.name);
                if (p && p->number_of_tokens() == 0) {
                    p->add_tokens(1);
                    missing += 1;
                }
            }
        }

        // now fire the transitio
        if (net.can_fire(*transition)) {
            // Store the current postfix-marking pair in the local cache
            MarkingPostfixKey key = {net.get_current_marking(), postfix};
            local_suffix_cache[key] = std::vector<std::string>();

            // add the to all firing sequences in the local cache 
            for (auto& [key, firing_sequence] : local_suffix_cache) {
                firing_sequence.push_back(event.activity);
            }
            net.fire_transition(*transition, &consumed, &produced);

        }else {
            throw std::runtime_error("Transition cannot be fired: " + event.activity);
        }
    }

    // Add the local suffix cache to the global suffix cache
    for (const auto& [key, value] : local_suffix_cache) {
        suffix_cache[key] = value;
    }

    consumed += net.final_marking.number_of_tokens();

    // Finalize the tokens in the Petri net
    finalize_tokens(net, silent_firing_sequences, missing, consumed, produced);

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
replay_trace_with_prefix_and_suffix(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache,
    std::unordered_map<std::string, std::tuple<std::tuple<int, int, int, int>, Marking>>& prefix_cache) {

    int missing = 0;   // Count of missing tokens (tokens added to input places to enable transitions)
    int remaining = 0; // Count of remaining tokens in the Petri net at the end
    int consumed = 0;  // Count of tokens consumed from input places
    int produced = 0;  // Count of tokens produced in output places

    return std::make_tuple(
        static_cast<double>(missing), 
        static_cast<double>(remaining), 
        static_cast<double>(produced), 
        static_cast<double>(consumed));
}


double calculate_fitness(const EventLog& log, const PetriNet& net, bool prefix_caching, bool suffix_caching){
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    // Map to store computed values for unique traces
    std::unordered_map<Trace, std::tuple<int, int, int, int>> trace_cache; 

    PetriNet net_copy = net;
    // A map to store the firing sequences for every place to every other place using silent transitions
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> silent_firing_sequences;
    silent_firing_sequences = get_places_shortest_path_by_hidden(net_copy, 5);

    // Activity cache to store the precomputed values
    ActivityCache activity_cache;

    // map of prefix and Marking + tuple with missing, remaining, produced, consumed
    std::unordered_map<std::string, std::tuple<std::tuple<int, int, int, int>, Marking>> prefix_cache;
    
    // something suffix
    std::unordered_map<MarkingPostfixKey, std::vector<std::string>, MarkingPostfixKeyHasher> suffix_cache;

    // Iterate over the traces in the event log
    for (const auto& trace : log.traces) {
        if (trace_cache.find(trace) == trace_cache.end()) {
            // If this trace has not been processed, do token replay
            PetriNet net_copy = net;
            if (prefix_caching && suffix_caching) {
                trace_cache[trace] = replay_trace_with_prefix_and_suffix(trace, net_copy, silent_firing_sequences, activity_cache , prefix_cache);
            } else if (prefix_caching) {
                trace_cache[trace] = replay_trace_with_prefix(trace, net_copy, silent_firing_sequences, activity_cache,  prefix_cache);
            } else if (suffix_caching) {
                trace_cache[trace] = replay_trace_with_suffix(trace, net_copy, silent_firing_sequences, activity_cache, suffix_cache);
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








