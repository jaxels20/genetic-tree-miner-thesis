
#include <gtest/gtest.h>
#include <set>
#include "Marking.hpp"
#include "silent_transition_handling.cpp"


// Test compute_delta_set function
TEST(MarkingTest, ComputeDeltaSet) {
    Marking current({{"p1", 2}, {"p2", 3}});
    Marking target({{"p1", 3}, {"p2", 3}, {"p3", 1}});

    std::set<std::string> expected_delta = {"p1", "p3"};
    EXPECT_EQ(compute_delta_set(current, target), expected_delta);
}

// Test compute_lambda_set function
TEST(MarkingTest, ComputeLambdaSet) {
    Marking current({{"p1", 3}, {"p2", 5}, {"p4", 1}});
    Marking target({{"p1", 3}, {"p2", 2}, {"p3", 1}});


    std::set<std::string> expected_lambda = {"p2", "p4"};
    EXPECT_EQ(compute_lambda_set(current, target), expected_lambda);
}

// Test compute_delta_set when no places need modification
TEST(MarkingTest, ComputeDeltaSet_NoChange) {
    Marking current({{"p1", 3}, {"p2", 3}});
    Marking target({{"p1", 3}, {"p2", 3}});

    std::set<std::string> expected_delta = {};
    EXPECT_EQ(compute_delta_set(current, target), expected_delta);
}

// Test compute_lambda_set when no places need modification
TEST(MarkingTest, ComputeLambdaSet_NoChange) {
    Marking current({{"p1", 3}, {"p2", 3}});
    Marking target({{"p1", 3}, {"p2", 3}});

    std::set<std::string> expected_lambda = {};
    EXPECT_EQ(compute_lambda_set(current, target), expected_lambda);
}

// Edge Case: Empty markings
TEST(MarkingTest, EmptyMarkings) {
    Marking current({});
    Marking target({});

    EXPECT_EQ(compute_delta_set(current, target), std::set<std::string>{});
    EXPECT_EQ(compute_lambda_set(current, target), std::set<std::string>{});
}

// Test compute_delta_set
TEST(MarkingTest, compute_delta_set) {
    Marking current({{"p1", 1}, {"p2", 2}});
    Marking target({{"p1", 2}, {"p3", 1}});
    
    std::set<std::string> expected = {"p1", "p3"};
    EXPECT_EQ(compute_delta_set(current, target), expected);
}

// Test compute_lambda_set
TEST(MarkingTest, compute_lambda_set) {
    Marking current({{"p1", 3}, {"p2", 2}});
    Marking target({{"p1", 1}, {"p2", 2}});
    
    std::set<std::string> expected = {"p1"};
    EXPECT_EQ(compute_lambda_set(current, target), expected);
}

// Test get_possible_firing_sequences
TEST(MarkingTest, get_possible_firing_sequences) {
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> firing_sequences = {
        {"p1", {{"p2", {"A", "B"}}}},
        {"p3", {{"p4", {"C"}}}}
    };
    
    std::set<std::string> delta_set = {"p2"};
    std::set<std::string> lambda_set = {"p1"};

    std::set<std::vector<std::string>, CompareVectorLength> expected = {{"A", "B"}};
    EXPECT_EQ(get_possible_firing_sequences(firing_sequences, delta_set, lambda_set), expected);
}

// Test Graph shortest path
TEST(MarkingTest, ShortestPathTest) {
    Graph g;
    g.addEdge("p1", "p2", 1, "A");
    g.addEdge("p2", "p3", 2, "tau_1");
    g.addEdge("p3", "p4", 1, "C");

    std::vector<std::string> expected = {"A", "tau_1", "C"};
    std::vector<std::string> result;
    
    ASSERT_TRUE(g.findShortestPath("p1", "p4", result));
    EXPECT_EQ(result, expected);
}

// Test computeAllPairsShortestPaths
TEST(MarkingTest, ComputeAllPairsTest) {
    Graph g;
    g.addEdge("p1", "p2", 1, "A");
    g.addEdge("p2", "p3", 2, "tau_1");
    g.addEdge("p3", "p4", 1, "C");

    auto paths = g.computeAllPairsShortestPaths();
    EXPECT_EQ(paths["p1"]["p4"], std::vector<std::string>({"A", "tau_1", "C"}));
}


TEST(MarkingTest, EnableTransitionFromSilentTransition){
    PetriNet net;

    net.add_place(Place("start", 1));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau_2"));
    net.add_transition(Transition("A"));

    net.add_arc(Arc("start", "tau_1", 1));
    net.add_arc(Arc("tau_1", "p1", 1));
    net.add_arc(Arc("p1", "tau_2", 1));
    net.add_arc(Arc("tau_2", "p2", 1));
    net.add_arc(Arc("p2", "A", 1));
    net.add_arc(Arc("A", "end", 1));

    Marking current({{"start", 1}});
    Marking target({{"end", 1}});

    net.set_initial_marking(current);
    net.set_final_marking(target);


    Graph silent_graph = create_silent_graph(net);

    auto silent_firing_sequences = silent_graph.computeAllPairsShortestPaths();

    auto [reachable, sequence] = attempt_to_make_transition_enabled_by_firing_silent_transitions(net, net.get_transition("A"), silent_firing_sequences);

    EXPECT_TRUE(reachable);
    EXPECT_EQ(sequence, std::vector<std::string>({"tau_1", "tau_2"}));



}