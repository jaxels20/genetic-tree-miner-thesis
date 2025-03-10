#include "gtest/gtest.h"
#include "PrefixTree.hpp"
#include "Eventlog.hpp"

TEST(PrefixTreeTest, CreateNodeTest) {
    PrefixTree tree;
    std::vector<Event> events = {
        Event("A", "timestamp1", {}),
        Event("B", "timestamp2", {})
    };
    auto node = tree.get_or_create_node(events);
    ASSERT_NE(node, nullptr);

    // Verify that the nodes exist along the tree from the root.
    auto nodeA = tree.root->children["A"];
    ASSERT_NE(nodeA, nullptr);
    auto nodeB = nodeA->children["B"];
    ASSERT_NE(nodeB, nullptr);
}

TEST(PrefixTreeTest, LongestMatchingPrefixTest) {
    PrefixTree tree;
    // Create a known path in the tree: A -> B -> C
    std::vector<Event> events = {
        Event("A", "t1", {}),
        Event("B", "t2", {}),
        Event("C", "t3", {})
    };
    tree.get_or_create_node(events);

    // Now query with a sequence that extends beyond the tree: A, B, C, D
    std::vector<Event> query = {
        Event("A", "t1", {}),
        Event("B", "t2", {}),
        Event("C", "t3", {}),
        Event("D", "t4", {}) // This event doesn't exist in the tree
    };

    std::vector<Event> matched_prefix;
    auto node = tree.get_longest_matching_prefix(query, matched_prefix);

    // The matched prefix should contain A, B, C
    ASSERT_EQ(matched_prefix.size(), 3);
    EXPECT_EQ(matched_prefix[0].activity, "A");
    EXPECT_EQ(matched_prefix[1].activity, "B");
    EXPECT_EQ(matched_prefix[2].activity, "C");
}

TEST(PrefixTreeTest, NoMatchingPrefixTest) {
    PrefixTree tree;
    std::vector<Event> query = {
        Event("X", "t1", {})
    };
    std::vector<Event> matched_prefix;
    auto node = tree.get_longest_matching_prefix(query, matched_prefix);

    // Since there is no matching child for "X", matched_prefix should be empty.
    ASSERT_TRUE(matched_prefix.empty());
    // The returned node should be the root.
    EXPECT_EQ(node, tree.root);
}

TEST(PrefixTreeTest, RepeatedCreateNodeDoesNotDuplicate) {
    PrefixTree tree;
    std::vector<Event> events = {
        Event("A", "t1", {}),
        Event("B", "t2", {})
    };

    auto node1 = tree.get_or_create_node(events);
    auto node2 = tree.get_or_create_node(events);
    // The two returned nodes should be identical.
    EXPECT_EQ(node1, node2);
}

TEST(PrefixTreeTest, NodeMarkingDefaultTest) {
    PrefixTree tree;
    std::vector<Event> events = { Event("A", "t1", {}) };
    auto node = tree.get_or_create_node(events);
    // The marking should be default constructed; thus, the number of tokens should be 0.
    EXPECT_EQ(node->marking.number_of_tokens(), 0);
    // Now add some tokens to the marking.
    node->marking.add_place("p1", 5);
    EXPECT_EQ(node->marking.number_of_tokens(), 5);
}
