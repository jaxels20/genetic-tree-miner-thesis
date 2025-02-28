


#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "Graph.hpp"


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

void finalize_tokens(PetriNet& net, Graph& silent_graph, int& missing, int& consumed, int& produced) {
    // Check if there are tokens in the final marking
    // If not, try to use silent transitions before adding tokens manually

    for (const auto& [place, tokens] : net.final_marking.places) {
        Place* p = net.get_place(place);
        if (!p) continue;

        int32_t tokens_in_place = p->number_of_tokens();
        if (tokens_in_place < tokens) {
            int32_t missing_tokens = tokens - tokens_in_place;

            // Attempt to fire silent transitions to reach the final marking
            Marking curr_marking = net.get_current_marking();
            for (const auto& [place, tokens] : curr_marking.places) {
                std::vector<std::string> path;
                bool path_exists = silent_graph.findShortestPath(place, p->name, path);
                if ( path_exists ) {
                    std::vector<std::string> transition_sequence = silent_graph.getTransitionSequence(path);
                    net.fire_transition_sequence(transition_sequence, &consumed, &produced);
                    break;
                }
            }
            // Check again if the place now has enough tokens
            tokens_in_place = p->number_of_tokens();
            if (tokens_in_place < tokens) {
                // If silent transitions couldn't generate enough tokens, add manually
                p->add_tokens(tokens - tokens_in_place);
                missing += tokens - tokens_in_place;
            }
        }
    }
}

std::tuple<double, double, double, double> replay_trace(const Trace& trace, PetriNet& net, Graph silent_graph) {
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
            for (const auto& input_place : input_places) {
                Marking curr_marking = net.get_current_marking();
                for (const auto& [place, tokens] : curr_marking.places) {
                    std::vector<std::string> path;
                    bool path_exists = silent_graph.findShortestPath(place, input_place.name, path);
                    if (path_exists){
                        std::vector<std::string> transition_sequence = silent_graph.getTransitionSequence(path);
                        
                        net.fire_transition_sequence(transition_sequence, &consumed, &produced);
                        if(net.can_fire(*transition)) {
                            break;
                        }
                    }
                }

                if (net.can_fire(*transition)) {
                    break;
                }
            }

            if(net.can_fire(*transition)) {
                net.fire_transition(*transition, &consumed, &produced);
                continue;
            }


           // Create a token in the pre-set of the transition
            for (const auto& place : net.get_preset(*transition)) {
                Place* p = net.get_place(place.name);
                if (p) {
                    p->add_tokens(1);
                    //produced += 1;
                    missing += 1;
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
    std::cout << "Missing: " << missing << std::endl;
    std::cout << "Remaining: " << remaining << std::endl;
    std::cout << "Produced: " << produced << std::endl;
    std::cout << "Consumed: " << consumed << std::endl;

    return std::make_tuple(static_cast<double>(missing), static_cast<double>(remaining), static_cast<double>(produced), static_cast<double>(consumed));
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

std::tuple<double, double> calculate_fitness_and_precision(const EventLog& log, const PetriNet& net) {
    int total_missing = 0;
    int total_remaining = 0;
    int total_produced = 0;
    int total_consumed = 0;

    Graph silent_graph = create_silent_graph(net);

    // Iterate over the traces in the event log
    for (const auto& trace : log.traces) {
        // copy the net
        PetriNet net_copy = net;
        // Replay the trace on the Petri net
        auto [missing, remaining, produced, consumed] = replay_trace(trace, net_copy, silent_graph);

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




