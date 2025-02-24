#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"

#include <vector>

// Example test case
TEST(FastTokenBasedReplayTest, SimpleSequence) {
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

    auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

    
    EXPECT_EQ(fitness, 1.0);
    EXPECT_EQ(precision, 1.0);

}

// // Example test case
TEST(FastTokenBasedReplayTest, SimpleLoop) {
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

    std::vector<std::string> trace_list = {"ABBBC", "ABBBC", "ABC", "AC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

    
    EXPECT_EQ(fitness, 1.0);
    EXPECT_EQ(precision, 1.0);

}

// Example test case
TEST(FastTokenBasedReplayTest, create_silent_graph) {
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

    std::vector<std::string> expected_nodes = {"p2", "p1"};
    std::vector<std::string> expected_edges = {"p1->p2"};


    std::vector<std::string> nodes = silent_graph.get_nodes();
    std::vector<std::string> edges = silent_graph.get_edges();


    EXPECT_EQ(nodes, expected_nodes);
    EXPECT_EQ(edges, expected_edges);


}

// Example test case
TEST(FastTokenBasedReplayTest, test_find_path_true) {
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
TEST(FastTokenBasedReplayTest, test_find_path_false) {
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
TEST(FastTokenBasedReplayTest, test_get_transition_sequence) {
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

// Example test case
TEST(FastTokenBasedReplayTest, test_simple_sequence_with_silent_transitions) {
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

    auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

    
    EXPECT_EQ(fitness, 1.0);
    EXPECT_EQ(precision, 1.0);

}

// Example test case
TEST(FastTokenBasedReplayTest, test_complex_with_silent_transitions) {
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

    auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

    
    EXPECT_EQ(fitness, 1.0);
    EXPECT_EQ(precision, 1.0);

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}