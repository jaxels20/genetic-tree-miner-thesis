#include <gtest/gtest.h>
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "token_based_replay.hpp"

#include <vector>

// Example test case
TEST(FastTokenBasedReplayTest, ExampleTest) {
    PetriNet net;
    net.add_place(Place("start", 1));
    net.add_place(Place("p1", 0));
    net.add_place(Place("p2", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("A", "p2"));
    net.add_arc(Arc("p1", "B"));
    net.add_arc(Arc("p2", "C"));
    net.add_arc(Arc("B", "end"));
    net.add_arc(Arc("C", "end"));

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