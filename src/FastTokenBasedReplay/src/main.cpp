#include "token_based_replay.cpp"
#include "precision.cpp"
#include "PetriNet.hpp"
#include "Eventlog.hpp"

int main() {

    PetriNet net;
    net.add_place(Place("start", 1));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("p3", 0));
    net.add_place(Place("p4", 0));
    net.add_place(Place("p5", 0));
    net.add_place(Place("p6", 0));
    net.add_place(Place("p7", 0));
    net.add_place(Place("p8", 0));
    net.add_place(Place("p9", 0));
    net.add_place(Place("p10", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));
    net.add_transition(Transition("D"));
    net.add_transition(Transition("E"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau_2"));
    net.add_transition(Transition("tau_3"));
    net.add_transition(Transition("tau_4"));
    net.add_transition(Transition("tau_5"));
    net.add_transition(Transition("tau_6"));

    net.add_arc(Arc("start", "tau_1"));
    net.add_arc(Arc("tau_1", "p1"));

    net.add_arc(Arc("p1", "C"));
    net.add_arc(Arc("p1", "tau_2"));
    net.add_arc(Arc("p1", "tau_3"));

    net.add_arc(Arc("tau_2", "p2"));
    net.add_arc(Arc("tau_2", "p3"));
    net.add_arc(Arc("p2", "B"));
    net.add_arc(Arc("p3", "A"));
    net.add_arc(Arc("B", "p4"));
    net.add_arc(Arc("A", "p5"));

    net.add_arc(Arc("tau_3", "p6"));
    net.add_arc(Arc("tau_3", "p7"));
    net.add_arc(Arc("p6", "D"));
    net.add_arc(Arc("p7", "E"));
    net.add_arc(Arc("D", "p8"));
    net.add_arc(Arc("E", "p9"));

    net.add_arc(Arc("p4", "tau_4"));
    net.add_arc(Arc("p5", "tau_4"));
    net.add_arc(Arc("tau_4", "p10"));

    net.add_arc(Arc("p8", "tau_5"));
    net.add_arc(Arc("p9", "tau_5"));
    net.add_arc(Arc("tau_5", "p10"));

    net.add_arc(Arc("C", "p10"));

    net.add_arc(Arc("p10", "tau_6"));
    net.add_arc(Arc("tau_6", "end"));



    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::set<std::string> expected = {"A"};
    std::set<std::string> result = net.get_visible_transitions_eventually_enabled();

    std::cout << "Result: ";
    for (const auto& task : result) {
        std::cout << task << " ";
    }



    return 0;
}