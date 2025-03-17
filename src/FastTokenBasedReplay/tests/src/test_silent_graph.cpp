#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"
#include "Graph.hpp"
#include "silent_transition_handling.cpp"

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

    std::vector<std::string> expected_path = {"tau_1", "tau2"};

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


TEST(new_silent_transition_finding, test1){
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("tau_1"));
    net.add_transition(Transition("tau_2"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "tau_1"));
    net.add_arc(Arc("tau_1", "p2"));
    net.add_arc(Arc("p2", "tau_2"));
    net.add_arc(Arc("tau_2", "end"));

    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));
    int max_rec_depth = 5;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> shortest_paths = get_places_shortest_path_by_hidden(net, max_rec_depth);

    EXPECT_EQ(shortest_paths["p1"]["end"].size(), 2);
    EXPECT_EQ(shortest_paths["p1"]["end"][0], "tau_1");
    EXPECT_EQ(shortest_paths["p1"]["end"][1], "tau_2");

    EXPECT_EQ(shortest_paths["p2"]["end"].size(), 1);
    EXPECT_EQ(shortest_paths["p2"]["end"][0], "tau_2");



}


class GraphTest : public ::testing::Test {
    protected:
        Graph graph;
    
        void SetUp() override {
            graph.addEdge("A", "B", 1, "t1");
            graph.addEdge("B", "C", 1, "t2");
            graph.addEdge("A", "C", 1, "t3");
            graph.addEdge("C", "D", 1, "t4");
        }
    };
    
    TEST_F(GraphTest, AddEdge) {
        EXPECT_EQ(graph.adjList["A"].size(), 2);
        EXPECT_EQ(graph.adjList["B"].size(), 1);
        EXPECT_EQ(graph.adjList["C"].size(), 1);
        EXPECT_EQ(graph.adjList["D"].size(), 0);
        EXPECT_EQ(graph.adjList["A"][0].destination, "B");
        EXPECT_EQ(graph.adjList["A"][0].weight, 1);
        EXPECT_EQ(graph.adjList["A"][0].transitionName, "t1");
    }
    
    TEST_F(GraphTest, FindShortestPath) {
        std::vector<std::string> firingSequence;
        bool pathExists = graph.findShortestPath("A", "D", firingSequence);
        EXPECT_TRUE(pathExists);
        ASSERT_EQ(firingSequence.size(), 2);
        EXPECT_EQ(firingSequence[0], "t3");
        EXPECT_EQ(firingSequence[1], "t4");
    }
    
    TEST_F(GraphTest, ComputeAllPairsShortestPaths) {
        auto shortestPaths = graph.computeAllPairsShortestPaths();
        EXPECT_EQ(shortestPaths["A"]["D"].size(), 2);
        EXPECT_EQ(shortestPaths["A"]["D"][0], "t3");
        EXPECT_EQ(shortestPaths["A"]["D"][1], "t4");
    }

