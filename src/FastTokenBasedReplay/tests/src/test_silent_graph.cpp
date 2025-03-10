#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"
#include "Graph.hpp"

// Example test case
TEST(SilentGraph, test_find_path_true) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau2"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "tau_1"));
    net.add_arc(Arc("tau_1", "p2"));
    net.add_arc(Arc("p2", "tau2"));
    net.add_arc(Arc("tau2", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    Graph silent_graph = create_silent_graph(net);

    std::vector<std::string> path;
    bool path_exists = silent_graph.findShortestPath("p1", "end", path);

    std::vector<std::string> expected_path = {"p1", "p2", "end"};

    EXPECT_EQ(path_exists, true);
    EXPECT_EQ(path, expected_path);

}

// Example test case
TEST(SilentGraph, test_find_path_false) {
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

    Graph silent_graph = create_silent_graph(net);

    std::vector<std::string> path;
    bool path_exists = silent_graph.findShortestPath("p1", "end", path);

    EXPECT_EQ(path_exists, false);

}

// Example test case
TEST(SilentGraph, test_get_transition_sequence) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau2"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "tau_1"));
    net.add_arc(Arc("tau_1", "p2"));
    net.add_arc(Arc("p2", "tau2"));
    net.add_arc(Arc("tau2", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    Graph silent_graph = create_silent_graph(net);

    std::vector<std::string> path;
    bool path_exists = silent_graph.findShortestPath("p1", "end", path);

    std::vector<std::string> expected_transition_sequence = {"tau_1", "tau2"};

    std::vector<std::string> transition_sequence = silent_graph.getTransitionSequence(path);

    EXPECT_EQ(transition_sequence, expected_transition_sequence);
    

}
