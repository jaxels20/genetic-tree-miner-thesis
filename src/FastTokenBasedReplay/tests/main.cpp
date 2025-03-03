#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"
#include "Place.hpp"
#include "Marking.hpp"
#include <vector>
#include "HyperGraph.hpp"

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


// Example test case
TEST(FastTokenBasedReplayTest, silent_transition_in_before_end_place) {
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

    auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

    
    EXPECT_EQ(fitness, 0.8);
    EXPECT_EQ(precision, 1.0);

}

// Example test case
// TEST(FastTokenBasedReplayTest, seq_and) {
//     PetriNet net;
//     net.add_place(Place("start", 0));
//     net.add_place(Place("p1", 0));
//     net.add_place(Place("p2", 0));
//     net.add_place(Place("p3", 0));
//     net.add_place(Place("p4", 0));
//     net.add_place(Place("end", 0));

//     net.add_transition(Transition("A"));
//     net.add_transition(Transition("tau_1"));
//     net.add_transition(Transition("B"));
//     net.add_transition(Transition("tau_2"));

//     net.add_arc(Arc("start", "tau_1"));
//     net.add_arc(Arc("tau_1", "p1"));
//     net.add_arc(Arc("tau_1", "p2"));

//     net.add_arc(Arc("p1", "A"));
//     net.add_arc(Arc("p2", "B"));

//     net.add_arc(Arc("A", "p3"));
//     net.add_arc(Arc("B", "p4"));

//     net.add_arc(Arc("p3", "tau_2"));
//     net.add_arc(Arc("p4", "tau_2"));
//     net.add_arc(Arc("tau_2", "end"));

//     net.set_initial_marking(Marking({{"start", 1}}));
//     net.set_final_marking(Marking({{"end", 1}}));

//     std::vector<std::string> trace_list = { "AA"};
//     EventLog eventlog = EventLog::from_trace_list(trace_list);

//     auto [fitness, precision] = calculate_fitness_and_precision(eventlog, net);

//     EXPECT_EQ(fitness, 0.8);
//     EXPECT_EQ(precision, 1.0);
    

// }

TEST(FastTokenBasedReplayTest, final_marking_condition) {
    Marking final_marking = Marking({{"p1", 1}});

    Marking m1 = Marking({{"p1", 1}});
    Marking m2 = Marking({{"p1", 2}});
    Marking m3 = Marking({{"p1", 0}});
    Marking m4 = Marking({{"p1", 0}, {"p2", 1}});
    Marking m5 = Marking({{"p1", 1}, {"p2", 1}});

    EXPECT_EQ(stop_condition_final_marking(m1, final_marking), true);
    EXPECT_EQ(stop_condition_final_marking(m2, final_marking), true);
    EXPECT_EQ(stop_condition_final_marking(m3, final_marking), false);
    EXPECT_EQ(stop_condition_final_marking(m4, final_marking), false);
    EXPECT_EQ(stop_condition_final_marking(m5, final_marking), true);
}

// HYPERGRAPH TESTS

// Unit Tests
TEST(HyperGraphTest, AddNode) {
    HyperGraph hg;
    hg.addNode("A", 3);
    EXPECT_TRUE(hg.hasNode("A"));
    EXPECT_EQ(hg.getTokens("A"), 3);
}

TEST(HyperGraphTest, AddEdge) {
    HyperGraph hg;
    hg.addNode("A");
    hg.addNode("B");
    hg.addNode("C");
    hg.addEdge("E1", {"A"}, {"B", "C"});
    EXPECT_TRUE(hg.hasEdge("E1"));
    auto sources = hg.getEdgeSources("E1");
    auto targets = hg.getEdgeTargets("E1");
    EXPECT_EQ(sources.size(), 1);
    EXPECT_EQ(targets.size(), 2);
    EXPECT_TRUE(sources.find("A") != sources.end());
    EXPECT_TRUE(targets.find("B") != targets.end());
    EXPECT_TRUE(targets.find("C") != targets.end());
}

TEST(HyperGraphTest, ModifyTokens) {
    HyperGraph hg;
    hg.addNode("X", 5);
    hg.setTokens("X", 10);
    EXPECT_EQ(hg.getTokens("X"), 10);
}

TEST(HyperGraphTest, SetAndResetMarking) {
    HyperGraph hg;
    hg.addNode("A", 3);
    hg.addNode("B", 2);
    hg.setMarking({{"A", 5}, {"B", 0}});
    EXPECT_EQ(hg.getTokens("A"), 5);
    EXPECT_EQ(hg.getTokens("B"), 0);
    hg.resetMarking();
    EXPECT_EQ(hg.getTokens("A"), 0);
    EXPECT_EQ(hg.getTokens("B"), 0);
}

TEST(HyperGraphTest, CanReachTargetMarking) {
    HyperGraph hg;
    hg.addNode("A", 1);
    hg.addNode("B", 0);
    hg.addNode("C", 0);
    hg.addEdge("E1", {"A"}, {"B"});
    hg.addEdge("E2", {"B"}, {"C"});
    
    Marking start = Marking({{"A", 1}, {"B", 0}, {"C", 0}});
    Marking target1 = Marking({{"B", 1}});
    Marking target2 = Marking({{"C", 1}});
    
    auto [reachable1, _] = hg.canReachTargetMarking(start, target1);
    auto [reachable2, __] = hg.canReachTargetMarking(start, target2);

    EXPECT_TRUE(reachable1);
    EXPECT_TRUE(reachable2);

}

TEST(HyperGraphTest, HyperEdgeFiring) {
    HyperGraph hg;
    hg.addNode("A", 1);
    hg.addNode("B", 1);
    hg.addNode("C", 0);
    hg.addNode("D", 0);

    // Hyperedge requires both A and B as sources, and produces tokens in C and D
    hg.addEdge("E1", {"A", "B"}, {"C", "D"});

    Marking start = Marking({{"A", 1}, {"B", 1}});
    Marking target = Marking({{"C", 1}, {"D", 1}});
    auto [reachable, _] = hg.canReachTargetMarking(start, target);
    EXPECT_TRUE(reachable);
}

TEST(HyperGraphTest, FindFiringSequence) {
    HyperGraph hg;
    hg.addNode("A", 0);
    hg.addNode("B", 0);
    hg.addNode("C", 0);

    hg.addEdge("E1", {"A"}, {"B"});
    hg.addEdge("E2", {"B"}, {"C"});

    Marking start = Marking({{"A", 1}});
    Marking target = Marking({{"C", 1}});

    auto [reachable, sequence] = hg.canReachTargetMarking(start, target);
    
    EXPECT_TRUE(reachable);
    ASSERT_EQ(sequence.size(), 2);
    EXPECT_EQ(sequence[0], "E1");
    EXPECT_EQ(sequence[1], "E2");
}

// Test the create_silent_hyper_graph function
TEST(CreateSilentHyperGraphTest, ConvertsSilentTransitionsCorrectly) {
    // Create a simple Petri net with some places and transitions, including silent ones
    PetriNet petriNet;

    // Add places to the Petri net
    petriNet.add_place(Place("P1", 1));  // Place P1 with 1 token
    petriNet.add_place(Place("P2", 0));  // Place P2 with 0 tokens
    petriNet.add_place(Place("P3", 2));  // Place P3 with 2 tokens

    // Add transitions (one of which is silent)
    petriNet.add_transition(Transition("T1"));
    petriNet.add_transition(Transition("tauT2"));  // Silent transition (starts with "tau")

    // Add arcs (connections between places and transitions)
    petriNet.add_arc(Arc("P1", "T1", 1));  // Arc from P1 to T1
    petriNet.add_arc(Arc("T1", "P2", 1));  // Arc from T1 to P2
    petriNet.add_arc(Arc("P3", "tauT2", 1));  // Arc from P3 to silent transition tauT2
    petriNet.add_arc(Arc("tauT2", "P1", 1));  // Arc from tauT2 to P1

    // Create the silent graph using the function
    HyperGraph silentGraph = create_silent_hyper_graph(petriNet);

    // Assert that the hypergraph contains the correct nodes (places)
    ASSERT_TRUE(silentGraph.hasNode("P1"));
    ASSERT_TRUE(silentGraph.hasNode("P2"));
    ASSERT_TRUE(silentGraph.hasNode("P3"));

    // Assert that the hypergraph contains the silent transition as a hyperedge
    ASSERT_TRUE(silentGraph.hasEdge("tauT2"));

    // Assert that the sources and targets of the silent transition are correct
    auto sources = silentGraph.getEdgeSources("tauT2");
    auto targets = silentGraph.getEdgeTargets("tauT2");
    
    ASSERT_EQ(sources.size(), 1);
    ASSERT_EQ(targets.size(), 1);

    // The source should be P3, and the target should be P1
    ASSERT_TRUE(sources.find("P3") != sources.end());
    ASSERT_TRUE(targets.find("P1") != targets.end());

    // Check if other transitions (like T1) are not included as hyperedges
    ASSERT_FALSE(silentGraph.hasEdge("T1"));
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}