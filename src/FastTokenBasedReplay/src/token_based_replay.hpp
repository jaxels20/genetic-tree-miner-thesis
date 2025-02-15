


#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "PetriNet.hpp"
#include "Eventlog.hpp"

std::tuple<double, double, double, double> replay_trace(const Trace& trace, PetriNet& net) {
    int missing = 0;   // Count of missing tokens (tokens added to input places to enable transitions)
    int remaining = 0; // Count of remaining tokens in the Petri net at the end
    int consumed = 0;  // Count of tokens consumed from input places
    int produced = 0;  // Count of tokens produced in output places

    // Iterate over the events in the trace
    for (const auto& event : trace.events) {
        // Find the corresponding transition for the event
        Transition* transition = net.get_transition(event.activity);

        if (!transition) {
            continue;  // Skip if no transition is found for the event
        }

        // Check if the transition is enabled (can fire)
        if (!net.can_fire(*transition)) {
            // If not enabled, we need to add tokens to the input places to enable it
            // Go through all the input arcs for this transition and add tokens
            for (const auto& arc : net.arcs) {
                if (arc.target == transition->name) {
                    Place* place = net.get_place(arc.source);
                    if (place) {
                        // Add tokens to the input place to enable the transition
                        place->add_tokens(arc.weight);
                        missing += arc.weight;  // Track how many tokens were added
                    }
                }
            }
        }

        // Fire the transition (if it can fire after adding tokens, if necessary)
        if (net.can_fire(*transition)) {
            // Track consumed tokens from input places
            for (const auto& arc : net.arcs) {
                if (arc.target == transition->name) {
                    Place* place = net.get_place(arc.source);
                    if (place) {
                        place->remove_tokens(arc.weight);
                        consumed += arc.weight; // Track consumed tokens
                    }
                }
            }

            // Track produced tokens in output places
            for (const auto& arc : net.arcs) {
                if (arc.source == transition->name) {
                    Place* place = net.get_place(arc.target);
                    if (place) {
                        place->add_tokens(arc.weight);
                        produced += arc.weight; // Track produced tokens
                    }
                }
            }
        }
    }

    // Count remaining tokens in the Petri net after the trace is processed
    for (const auto& place : net.places) {
        remaining += place.tokens;
    }

    // print the results
    //std::cout << "Missing: " << missing << std::endl;
    //std::cout << "Remaining: " << remaining << std::endl;
    //std::cout << "Produced: " << produced << std::endl;
    //std::cout << "Consumed: " << consumed << std::endl;

    return std::make_tuple(static_cast<double>(missing), static_cast<double>(remaining), static_cast<double>(produced), static_cast<double>(consumed));
}


std::tuple<double, double> calculate_fitness_and_precision(const EventLog& log, const PetriNet& net) {
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    // copy the net
    PetriNet net_copy = net;

    // Iterate over the traces in the event log
    for (const auto& trace : log.traces) {
        // Replay the trace on the Petri net
        auto [missing, remaining, produced, consumed] = replay_trace(trace, net_copy);

        // Update the total counts
        total_missing += missing;
        total_remaining += remaining;
        total_produced += produced;
        total_consumed += consumed;
    }

    double fitness = 0.5 * (1 - (static_cast<double>(total_missing) / total_consumed)) + 0.5 * (1 - (static_cast<double>(total_remaining) / total_produced));
    


    return std::make_tuple(fitness, 1.0);
}

double get_percentage_of_fitting_traces(const EventLog& log, const PetriNet& net) {
    return 1.0;
}




