#include "token_based_replay.cpp"
#include "PetriNet.hpp"
#include "Eventlog.hpp"

int main() {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("p3", 0));
    net.add_place(Place("p4", 0));
    net.add_place(Place("p5", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau_2"));
    net.add_transition(Transition("tau_3"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("A", "p3"));

    net.add_arc(Arc("p3", "tau_1"));
    net.add_arc(Arc("tau_1", "p4"));
    
    net.add_arc(Arc("p1", "tau_2"));
    net.add_arc(Arc("tau_2", "p2"));

    net.add_arc(Arc("p2", "tau_3"));
    net.add_arc(Arc("p4", "tau_3"));
    net.add_arc(Arc("tau_3", "p5"));

    net.add_arc(Arc("p5", "B"));
    net.add_arc(Arc("B", "end"));


    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AB"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, false);

    std::cout << "Fitness: " << fitness << std::endl;
    //assert (fitness == 1);




    return 0;
}