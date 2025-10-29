#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "stdio.h"
#include <algorithm>
#include <set>
#include <map>
#include "disjoint_set.h"

Graph::Graph() : num_nodes(0), num_edges(0)
{
    edge_list.clear();
    adj_list.clear();
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        adj_list[i].clear();

    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);
        adj_list[y].push_back(x);
        edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
    }
}

Graph::~Graph()
{
    edge_list.clear();
    adj_list.clear();
    num_nodes = 0;
    num_edges = 0;
}

double Graph::getTwoRankNeighborsRatio(std::vector<int> covered)
{
    std::set<int> tempSet;
    for(int i =0;i<(int)covered.size();++i){
        tempSet.insert(covered[i]);
    }
    double sum  = 0;
    for(int i =0;i<num_nodes;++i){
        if(tempSet.count(i)==0){
        for(int j=i+1;j<num_nodes;++j){
        if(tempSet.count(j)==0){
            std::vector<int> v3;
            std::set_intersection(adj_list[i].begin(),adj_list[i].end(),adj_list[j].begin(),adj_list[j].end(),std::inserter(v3,v3.begin()));
            if(v3.size()>0){
                sum += 1.0;
            }
        }
        }
        }
    }
    return sum;
}

std::vector<std::vector<int>> Graph::getCcDescending(const std::set<int> covered) const
{
    // Use Union-Find for better performance on large graphs
    Disjoint_Set disjoint_set(num_nodes);
    
    // Build connected components by merging adjacent uncovered nodes
    for (int i = 0; i < num_nodes; i++) {
        if (covered.count(i) == 0) {  // Skip covered nodes
            for (auto neigh : adj_list[i]) {
                if (covered.count(neigh) == 0) {  // Skip covered neighbors
                    disjoint_set.merge(i, neigh);  // Union adjacent uncovered nodes
                }
            }
        }
    }
    
    // Group nodes by their root representative
    std::map<int, std::vector<int>> component_map;
    for (int i = 0; i < num_nodes; i++) {
        if (covered.count(i) == 0) {  // Only process uncovered nodes
            int root = disjoint_set.findRoot(i);
            component_map[root].push_back(i);
        }
    }
    
    // Convert map to vector and sort components by size (descending)
    std::vector<std::vector<int>> components;
    components.reserve(component_map.size());
    
    for (auto& pair : component_map) {
        if (!pair.second.empty()) {
            // Sort nodes within component for consistency
            std::sort(pair.second.begin(), pair.second.end());
            components.push_back(std::move(pair.second));
        }
    }
    
    // Sort components by size in descending order
    std::sort(components.begin(), components.end(), 
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  return a.size() > b.size();
              });
    
    return components;
}

std::vector<int> Graph::getCcSizesDescending(const std::set<int> covered) const
{
    // Ultra-fast version that only returns component sizes, not node lists
    // Perfect for cases where we only need sizes (like componentRollout)
    
    Disjoint_Set disjoint_set(num_nodes);
    
    // Build connected components by merging adjacent uncovered nodes
    for (int i = 0; i < num_nodes; i++) {
        if (covered.count(i) == 0) {  // Skip covered nodes
            for (auto neigh : adj_list[i]) {
                if (covered.count(neigh) == 0) {  // Skip covered neighbors
                    disjoint_set.merge(i, neigh);  // Union adjacent uncovered nodes
                }
            }
        }
    }
    
    // Collect component sizes using a map to avoid duplicates
    std::map<int, int> size_map;
    for (int i = 0; i < num_nodes; i++) {
        if (covered.count(i) == 0) {  // Only process uncovered nodes
            int root = disjoint_set.findRoot(i);
            size_map[root] = disjoint_set.getRank(root);
        }
    }
    
    // Convert to vector and sort by size (descending)
    std::vector<int> sizes;
    sizes.reserve(size_map.size());
    
    for (const auto& pair : size_map) {
        sizes.push_back(pair.second);
    }
    
    std::sort(sizes.begin(), sizes.end(), std::greater<int>());
    
    return sizes;
}

GSet::GSet()
{
    graph_pool.clear();
}

GSet::~GSet()
{
    graph_pool.clear();
}

void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
//    printf("graph_pool_size:%d",graph_pool.size());
    assert(graph_pool.size());
//    printf("graph_pool_size:%d",graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;