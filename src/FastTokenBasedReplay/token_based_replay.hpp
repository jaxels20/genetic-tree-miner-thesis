


#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "PetriNet.hpp"
#include "Eventlog.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include "PetriNet.hpp"    // Assume your PetriNet, Place, Transition, Arc classes are here
#include "Eventlog.hpp"    // Assume your Event, Trace, EventLog classes are here
#include "FitnessResult.hpp"  // Assume your FitnessResult struct is here

FitnessResult calculate_fitness(const EventLog& log, const PetriNet& net) {
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    // Process each trace in the log individually.
    for (const Trace& trace : log.traces) {
        // Create a copy of the initial marking (place tokens) from the Petri net.
        std::unordered_map<std::string, int> marking;
        for (const Place& place : net.places) {
            marking[place.name] = place.tokens;
        }

        // Replay each event in the trace.
        for (const Event& event : trace.events) {
            // Find the corresponding transition in the Petri net by matching the activity name.
            auto trans_it = std::find_if(net.transitions.begin(), net.transitions.end(),
                                         [&event](const Transition& t) { return t.name == event.activity; });
            if (trans_it == net.transitions.end()) {
                // If no matching transition is found, you might decide to ignore the event,
                // count it as a deviation, or handle it differently.
                std::cerr << "Warning: Transition '" << event.activity << "' not found in the Petri net.\n";
                continue;
            }
            const Transition& t = *trans_it;

            // Process input arcs: these are arcs where the transition is the target.
            // We assume that if an arc’s source is one of the Petri net’s places, it is an input arc.
            for (const Arc& arc : net.arcs) {
                if (arc.target == t.name && marking.find(arc.source) != marking.end()) {
                    int required = arc.weight;
                    int available = marking[arc.source];
                    if (available < required) {
                        int missingTokens = required - available;
                        total_missing += missingTokens;
                        // “Insert” the missing tokens into the marking.
                        marking[arc.source] += missingTokens;
                    }
                    // Now consume the required tokens.
                    marking[arc.source] -= required;
                    total_consumed += required;
                }
            }

            // Process output arcs: these are arcs where the transition is the source.
            // We assume that if an arc’s target is a place in the Petri net, it is an output arc.
            for (const Arc& arc : net.arcs) {
                if (arc.source == t.name && marking.find(arc.target) != marking.end()) {
                    marking[arc.target] += arc.weight;
                    total_produced += arc.weight;
                }
            }
        }

        // After replaying a trace, the tokens left in all places are considered "remaining".
        for (const auto& kv : marking) {
            total_remaining += kv.second;
        }
    }

    // Calculate fitness using the formula:
    // fitness = 1 - (missing + remaining) / (produced + consumed)
    double fitness = 1.0;
    int denominator = total_produced + total_consumed;
    if (denominator != 0) {
        fitness = 1.0 - static_cast<double>(total_missing + total_remaining) / denominator;
    }
    
    return FitnessResult{fitness, total_missing, total_remaining, total_produced, total_consumed};
}


