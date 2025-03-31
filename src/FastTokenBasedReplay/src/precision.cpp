#pragma once
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "unordered_map"
#include "set"
#include "string"
#include "vector"
#include "tuple"
#include "silent_transition_handling.cpp"
#include "ActivityCache.hpp"
#include "token_based_replay.cpp"


std::unordered_map<std::string, std::set<std::string>> compute_prefixes(const EventLog& log) {
    std::unordered_map<std::string, std::set<std::string>> prefixes;

    for (const auto& trace : log.traces) {
        std::string prefix;
        // Remember to add the empty prefix with the first activity
        prefixes[prefix].insert(trace.events[0].activity);

        for (int i = 0; i < trace.events.size(); ++i) {
            std::string curr_event = trace.events[i].activity;
            std::string next_event = (i + 1 < trace.events.size()) ? trace.events[i + 1].activity : "";
            
            prefix += curr_event + ",";
            prefixes[prefix].insert(next_event);
            
        }
    }
    
    return prefixes;
}


std::tuple<int32_t, int32_t> replay_trace_precision(
    const Trace& trace, 
    PetriNet& net, 
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<std::string>>> silent_firing_sequences,
    ActivityCache& activity_cache,
    std::unordered_map<std::string, std::set<std::string>> prefixes){
    int32_t escaped_edges = 0;
    int32_t allowed_tasks = 0;
    std::string current_prefix;

    // Initialize the tokens in the Petri net
    initialize_tokens(net);

    // Iterate over the events in the trace
    for (const auto& event : trace.events) {
        
        // Find the transition corresponding to the event
        Transition* transition = net.get_transition(event.activity);
        if (!transition) {
            throw std::runtime_error("Transition not found: " + event.activity);
        }

        // DO THE BOOKKEEPING
        // Count the number of allowed tasks, which is the number of enabled transitions
        PetriNet net_copy = net;
        std::set<std::string> allowed_tasks_set = net_copy.get_visible_transitions_eventually_enabled();

        // print the allowed tasks
        std::cout << "Allowed tasks: ";
        for (const auto& task : allowed_tasks_set) {
            std::cout << task << " ";
        }
        std::cout << std::endl;

        allowed_tasks += allowed_tasks_set.size();

        std::set next_activity_after_prefix = prefixes[current_prefix];

        // Store result
        std::vector<std::string> difference_result;
        std::set_difference(
            allowed_tasks_set.begin(), allowed_tasks_set.end(),
            next_activity_after_prefix.begin(), next_activity_after_prefix.end(),
            std::back_inserter(difference_result)
        );

        // Count the number of escaped edges
        escaped_edges += difference_result.size();

        // if the transition is not enabled, we need to fire silent transitions to make it enabled
        if (!net.can_fire(*transition)) {
            Marking current_marking = net.get_current_marking();
            std::vector<std::string> cached_sequence = activity_cache.retrieve(current_marking, transition->name);
            
            if (!cached_sequence.empty()) {
                net.fire_transition_sequence(cached_sequence, nullptr, nullptr);
            } else {
                auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, transition, silent_firing_sequences);     
                if (reachable) {
                    net.fire_transition_sequence(sequence, nullptr, nullptr);
                    activity_cache.store(current_marking, transition->name, sequence);                    
                } 
            }
        }

        // Now we can fire the transition (if it is enabled)
        if (net.can_fire(*transition)) {
            // count the number of escaped edges, which is the current prefix in the map differenced from the allowed tasks
            net.fire_transition(*transition, nullptr, nullptr);

            // update the current prefix
            current_prefix += event.activity + ",";

        } else {
            // For the precision we stop of the trace cannot be replayed
            break;
        }

    }

    return std::make_tuple(escaped_edges, allowed_tasks);
}




double calculate_precision(const EventLog& log, const PetriNet& net){

    auto prefixes = compute_prefixes(log);
    double precision = 0.0;
    int32_t total_escaping_edges = 0;
    int32_t total_allowed_tasks = 0;

    // Map to store computed values for unique traces
    std::unordered_map<Trace, std::tuple<int32_t, int32_t>> trace_cache;

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
            trace_cache[trace] = replay_trace_precision(trace, net_copy, silent_firing_sequences, activity_cache, prefixes);
        }

        // Get the replay result for the trace
        std::tuple<int, int> replay_result = trace_cache[trace];
        int32_t escaped_edges = std::get<0>(replay_result);
        int32_t allowed_tasks = std::get<1>(replay_result);
        total_escaping_edges += escaped_edges;
        total_allowed_tasks += allowed_tasks;
    }
    std::cout << "Total escaping edges: " << total_escaping_edges << std::endl;
    std::cout << "Total allowed tasks: " << total_allowed_tasks << std::endl;

    if (total_allowed_tasks == 0) {
        return 0.0;
    }
    precision = 1.0 - static_cast<double>(total_escaping_edges) / static_cast<double>(total_allowed_tasks);
    return precision;

    
}

