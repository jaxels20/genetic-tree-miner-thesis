#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"

TEST(NoCaching, SimpleSequence) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "B"));
    net.add_arc(Arc("B", "p2"));
    net.add_arc(Arc("p2", "C"));
    net.add_arc(Arc("C", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"ABC", "ABC", "ABC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, false);

    EXPECT_EQ(fitness, 1.0);
}

TEST(NoCaching, SimpleLoop) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "B"));
    net.add_arc(Arc("B", "p1"));
    net.add_arc(Arc("p1", "C"));
    net.add_arc(Arc("C", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AC", "ABC", "ABBC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);



    double fitness = calculate_fitness(eventlog, net, false, false);

    EXPECT_EQ(fitness, 1.0);
}

// Example test case
TEST(NoCaching, test_simple_sequence_with_silent_transitions) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "tau_1"));
    net.add_arc(Arc("tau_1", "p2"));
    net.add_arc(Arc("p2", "C"));
    net.add_arc(Arc("C", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);
    double fitness = calculate_fitness(eventlog, net, false, false);

    
    EXPECT_EQ(fitness, 1.0);

}

// Example test case
TEST(NoCaching, test_complex_with_silent_transitions) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("p3", 0));
    net.add_place(Place("p4", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("A", "p2"));

    net.add_arc(Arc("p1", "tau_1"));
    net.add_arc(Arc("p2", "B"));

    net.add_arc(Arc("tau_1", "p3"));
    net.add_arc(Arc("B", "p4"));

    net.add_arc(Arc("p3", "C"));
    net.add_arc(Arc("p4", "C"));

    net.add_arc(Arc("C", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"ABC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, false);

    
    EXPECT_EQ(fitness, 1.0);

}

// Example test case
TEST(NoCaching, silent_transition_in_before_end_place) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("tau_2"));

    net.add_arc(Arc("start", "tau_1"));
    net.add_arc(Arc("tau_1", "p1"));

    net.add_arc(Arc("p1", "A"));
    net.add_arc(Arc("p1", "B"));

    net.add_arc(Arc("B", "p2"));
    net.add_arc(Arc("A", "p2"));

    net.add_arc(Arc("p2", "tau_2"));
    net.add_arc(Arc("tau_2", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AB"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, false);

    
    EXPECT_EQ(fitness, 0.8);

}


// tricky final marking
TEST(NoCaching, test_tricky_final_marking) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("p3", 0));
    net.add_place(Place("p4", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
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
    net.add_arc(Arc("tau_3", "end"));


    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"A"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    double fitness = calculate_fitness(eventlog, net, false, false);

    
    EXPECT_EQ(fitness, 1);
}

TEST(NoCaching, test_tricky_enanbled_transition){
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

    EXPECT_EQ(fitness, 1);
}