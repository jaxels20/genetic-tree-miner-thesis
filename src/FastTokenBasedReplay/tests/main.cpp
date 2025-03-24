#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.cpp"
#include "Place.hpp"
#include "Marking.hpp"
#include <vector>
#include "HyperGraph.hpp"
#include "PrefixTree.hpp"
#include "SuffixTree.hpp"
#include <tuple>


// other tests
#include "src/test_prefix.cpp"
#include "src/test_no_caching.cpp"
#include "src/test_suffix.cpp"
#include "src/test_prefix_and_suffix.cpp"
#include "src/test_suffix_tree.cpp"
#include "src/test_prefix_tree.cpp"
#include "src/test_silent_graph.cpp"
#include "src/test_silent_transition_handling.cpp"

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

TEST(FastTokenBasedReplayTest, set_marking){
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

    Marking m = Marking({{"p1", 1}, {"p2", 1}});
    net.set_marking(m);

    Marking current_marking = net.get_current_marking();

    EXPECT_EQ(current_marking, m);
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}