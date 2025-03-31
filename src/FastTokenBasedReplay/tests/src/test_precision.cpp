
#include "precision.cpp"
#include "PetriNet.hpp"
#include "Eventlog.hpp"
#include "unordered_map"
#include "set"

#include "gtest/gtest.h"

TEST(Precision, compute_prefixes) {
    std::vector<std::string> trace_list = {"BC", "AC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    std::unordered_map<std::string, std::set<std::string>> result = compute_prefixes(eventlog);

    std::unordered_map<std::string, std::set<std::string>> expected = {
        {"B,", {"C"}},
        {"BC,", {}},
        {"A,", {"C"}},
        {"AC,", {}},
        {"", {"B", "A"}}
    };
    EXPECT_EQ(result.size(), expected.size());

    for (const auto& [prefix, activities] : expected) {
        EXPECT_EQ(result[prefix], activities);
    }

}


TEST(Precision, precision_simple) {
    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("C"));

    net.add_arc(Arc("start", "A"));
    net.add_arc(Arc("start", "B"));

    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("B", "p1"));
    
    net.add_arc(Arc("p1", "C"));
    net.add_arc(Arc("C", "end"));
    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::vector<std::string> trace_list = {"AC", "BC"};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    auto precision = calculate_precision(eventlog, net);
    EXPECT_EQ(precision, 1.0);

}
