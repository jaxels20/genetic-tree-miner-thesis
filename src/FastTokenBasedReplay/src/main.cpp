#include "token_based_replay.cpp"
#include "PetriNet.hpp"
#include "Eventlog.hpp"

int main() {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));


    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("start", "B"));

    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("B", "p1"));
    
    net.add_arc(Arc("p1", "C"));
    net.add_arc(Arc("C", "end"));


    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AC", "BC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, true);

    std::cout << "Fitness: " << fitness << std::endl;




    return 0;
}