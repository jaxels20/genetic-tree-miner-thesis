#pragma once
#include <vector>
#include <queue>
#include <climits>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "PetriNet.hpp"

class Graph {
    public:
        struct Edge {
            std::string destination;
            int weight;
            std::string transitionName;
        };
    
        std::unordered_map<std::string, std::vector<Edge>> adjList;
    
        void addEdge(const std::string& u, const std::string& v, int weight, const std::string& transitionName) {
            adjList[u].push_back({v, weight, transitionName});
            if (adjList.find(v) == adjList.end()) {
                adjList[v] = {};
            }
        }
        
        bool findShortestPath(const std::string& start, const std::string& end, std::vector<std::string>& firingSequence) {
            std::unordered_map<std::string, int> dist;
            std::unordered_map<std::string, std::string> parent;
            std::unordered_map<std::string, std::string> transitionUsed;
            std::priority_queue<std::pair<int, std::string>, std::vector<std::pair<int, std::string>>, std::greater<>> pq;

            for (const auto& node : adjList) {
                dist[node.first] = INT_MAX;
            }
            dist[start] = 0;
            pq.push({0, start});

            while (!pq.empty()) {
                std::string current = pq.top().second;
                pq.pop();

                if (current == end) {
                    std::string temp = end;
                    while (temp != start) {
                        firingSequence.push_back(transitionUsed[temp]);
                        temp = parent[temp];
                    }
                    std::reverse(firingSequence.begin(), firingSequence.end());
                    return true;
                }

                for (const auto& edge : adjList[current]) {
                    const std::string& nextNode = edge.destination;
                    int weight = edge.weight;

                    if (dist[current] + weight < dist[nextNode]) {
                        dist[nextNode] = dist[current] + weight;
                        parent[nextNode] = current;
                        transitionUsed[nextNode] = edge.transitionName;
                        pq.push({dist[nextNode], nextNode});
                    }
                }
            }
            return false;
        }

        std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>
        computeAllPairsShortestPaths() {
            std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> shortestPaths;
    
            for (const auto& source : adjList) {
                for (const auto& target : adjList) {
                    if (source.first != target.first) {
                        std::vector<std::string> firingSequence;
                        if (findShortestPath(source.first, target.first, firingSequence)) {
                            shortestPaths[source.first][target.first] = firingSequence;
                        }
                    }
                }
            }
            return shortestPaths;
        }
    };









