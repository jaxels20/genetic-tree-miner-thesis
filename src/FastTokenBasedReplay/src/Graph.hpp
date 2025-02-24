#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

class Graph {
public:
    // Struct to represent an edge with a transition name
    struct Edge {
        std::string destination;
        int weight;
        std::string transitionName;
    };

    // Adjacency list representation of the graph using string-based node names
    std::unordered_map<std::string, std::vector<Edge>> adjList;

    // Add an edge with a transition name
    void addEdge(const std::string& u, const std::string& v, int weight, const std::string& transitionName) {
        adjList[u].push_back({v, weight, transitionName}); // Directed edge from u to v
        if (adjList.find(v) == adjList.end()) {
            adjList[v] = {}; // Ensure v exists in the adjacency list
        }
    }

    // Finds the shortest path from start to end using Dijkstra's algorithm
    bool findShortestPath(const std::string& start, const std::string& end, std::vector<std::string>& path) {
        std::unordered_map<std::string, int> dist;
        std::unordered_map<std::string, std::string> parent;
        std::priority_queue<std::pair<int, std::string>, std::vector<std::pair<int, std::string>>, std::greater<>> pq;

        // Initialize distances
        for (const auto& node : adjList) {
            dist[node.first] = INT_MAX;
        }
        dist[start] = 0;

        pq.push({0, start});

        while (!pq.empty()) {
            std::string current = pq.top().second;
            pq.pop();

            if (current == end) {
                // Reconstruct the shortest path from end to start
                std::string temp = end;
                while (temp != start) {
                    path.push_back(temp);
                    temp = parent[temp];
                }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                return true;  // Path found
            }

            for (const auto& edge : adjList[current]) {
                const std::string& nextNode = edge.destination;
                int weight = edge.weight;

                if (dist[current] + weight < dist[nextNode]) {
                    dist[nextNode] = dist[current] + weight;
                    parent[nextNode] = current;
                    pq.push({dist[nextNode], nextNode});
                }
            }
        }

        return false; // No path found
    }

    // Get all the nodes in the graph
    std::vector<std::string> get_nodes() const {
        std::vector<std::string> nodes;
        for (const auto& node : adjList) {
            nodes.push_back(node.first);
        }
        return nodes;
    }

    // Get all the edges in the graph with transition names
    std::vector<std::string> get_edges() const {
        std::vector<std::string> edges;
        for (const auto& node : adjList) {
            for (const auto& edge : node.second) {
                edges.push_back(node.first + "->" + edge.destination);
            }
        }
        return edges;
    }

    // Get the sequence of transition names for a given path
    std::vector<std::string> getTransitionSequence(const std::vector<std::string>& path) const {
        std::vector<std::string> transitionSequence;

        for (size_t i = 0; i < path.size() - 1; ++i) {
            const std::string& current = path[i];
            const std::string& next = path[i + 1];

            // Find the corresponding edge with the transition name
            for (const auto& edge : adjList.at(current)) {
                if (edge.destination == next) {
                    transitionSequence.push_back(edge.transitionName);
                    break;
                }
            }
        }

        return transitionSequence;
    }
};


// int main() {
//     Graph g;

//     // Add directed edges to the graph (node1, node2, weight)
//     g.addEdge("A", "B", 10);
//     g.addEdge("A", "C", 5);
//     g.addEdge("B", "C", 2);
//     g.addEdge("B", "D", 1);
//     g.addEdge("C", "D", 9);
//     g.addEdge("D", "E", 4);
    
//     // Find path between nodes A and D
//     std::vector<std::string> path;
//     if (g.findPath("A", "D", path)) {
//         std::cout << "Path between A and D: ";
//         for (const auto& node : path) {
//             std::cout << node << " ";
//         }
//         std::cout << "\n";
//     } else {
//         std::cout << "No path found between A and D.\n";
//     }

//     // Find the shortest path between nodes A and E
//     path.clear();
//     int distance = g.findShortestPath("A", "E", path);
//     if (distance != -1) {
//         std::cout << "Shortest path between A and E (Distance: " << distance << "): ";
//         for (const auto& node : path) {
//             std::cout << node << " ";
//         }
//         std::cout << "\n";
//     } else {
//         std::cout << "No shortest path found between A and E.\n";
//     }

//     return 0;
// }
