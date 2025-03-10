#include <gtest/gtest.h>
#include "HyperGraph.hpp"
#include "PetriNet.hpp"
#include "token_based_replay.cpp"

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
    
    auto [reachable1, _] = hg.canReachTargetMarking(start, target1, 10);
    auto [reachable2, __] = hg.canReachTargetMarking(start, target2, 10);

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
    auto [reachable, _] = hg.canReachTargetMarking(start, target, 10);
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

    auto [reachable, sequence] = hg.canReachTargetMarking(start, target, 10);
    
    EXPECT_TRUE(reachable);
    ASSERT_EQ(sequence.size(), 2);
    EXPECT_EQ(sequence[0], "E1");
    EXPECT_EQ(sequence[1], "E2");
}

// Test the create_silent_hyper_graph function
TEST(HyperGraphTest, ConvertsSilentTransitionsCorrectly) {
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
