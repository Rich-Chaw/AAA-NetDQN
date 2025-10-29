#include "graph.h"
#include "mvc_env.h"
#include <iostream>
#include <memory>
#include <set>
using namespace std;

void test_helloworld(){
    cout << "Hello, World!" << endl;
}

bool test_graph(){
    // Test Graph class functionality
    cout << "\n=== Testing Graph class functions in graph.cpp ===" << endl;
    // create a simple graph for testing
    int num_nodes = 11;
    int num_edges = 10; 
    const int edges_from[10] = {0,0,1,1,2,3,4,5,6,7};
    const int edges_to[10] = {1,2,3,4,5,6,7,8,9,10};
    Graph g = Graph(num_nodes,num_edges,edges_from,edges_to);
    cout << "num_nodes: " << g.num_nodes << endl;
    cout << "num_edges: " << g.num_edges << endl;
    
    // Test connected components functionality
    cout << "\n=== Testing Connected Components ===" << endl;
    
    // Get all connected components (no exclusions)
    auto components = g.getCcDescending(std::set<int>());
    cout << "Total components: " << components.size() << endl;
    for (size_t i = 0; i < components.size(); i++) {
        cout << "Component " << i << " (size " << components[i].size() << "): ";
        for (int node : components[i]) {
            cout << node << " ";
        }
        cout << endl;
    }

    // Get connected components excluding certain nodes (simulating covered nodes)
    // 3 components: {3,6,9} and {4,7,10} and {0,2}
    std::set<int> covered_nodes = {1, 5, 8};
    auto components_after_covering = g.getCcDescending(covered_nodes);
    cout << "\nAfter covering nodes {1, 5, 8}:" << endl;
    cout << "Components: " << components_after_covering.size() << endl;
    for (size_t i = 0; i < components_after_covering.size(); i++) {
        cout << "Component " << i << " (size " << components_after_covering[i].size() << "): ";
        for (int node : components_after_covering[i]) {
            cout << node << " ";
        }
        cout << endl;
    }

    // Access the largest connected component
    if (!components_after_covering.empty()) {
        std::vector<int> largest_component = components_after_covering[0];
        int largest_size = largest_component.size();
        cout << "Largest component size: " << largest_size << endl;
    }
    
    return true;
}

bool test_mvc_env(){
    // Test Graph class functionality
    cout << "\n=== Testing MvcEnv class functions in mvc_env.cpp ===" << endl;

    cout << "\n=== Testing Union-Find Optimization ===" << endl;
    // Create a simple graph for testing
    int num_nodes = 8;
    int num_edges = 6;
    // Create a graph with two components: {0,1,2,3} and {4,5,6,7}
    const int edges_from[6] = {0,1,2,4,5,6};
    const int edges_to[6] = {1,2,3,5,6,7};
    
    std::shared_ptr<Graph> graph_ptr(new Graph(num_nodes, num_edges, edges_from, edges_to));
    
    // Test the new optimized component analysis
    cout << "Testing Union-Find vs DFS performance:" << endl;
    
    std::set<int> covered_nodes = {1, 5};
    
    // Test both methods
    auto components_dfs = graph_ptr->getCcDescending(covered_nodes);
    auto sizes_uf = graph_ptr->getCcSizesDescending(covered_nodes);
    
    cout << "DFS method - Components: " << components_dfs.size() << endl;
    for (size_t i = 0; i < components_dfs.size(); i++) {
        cout << "  Component " << i << " (size " << components_dfs[i].size() << "): ";
        for (int node : components_dfs[i]) {
            cout << node << " ";
        }
        cout << endl;
    }
    
    cout << "Union-Find method - Component sizes: ";
    for (int size : sizes_uf) {
        cout << size << " ";
    }
    cout << endl;
    
    cout << "\n=== Testing Component Rollout ===" << endl;
    
    // Scenario 1: Moderately fragmented graph
    MvcEnv env1(1.0);
    env1.s0(graph_ptr);
    env1.covered_set.insert(1);
    env1.covered_set.insert(5);
    
    cout << "Scenario 1 - Moderate fragmentation:" << endl;
    cout << "  Graph nodes: " << graph_ptr->num_nodes << endl;
    cout << "  Covered nodes: {1, 5}" << endl;
    cout << "  LCC size: " << env1.getMaxConnectedNodesNum() << endl;
    
    auto sizes1 = env1.getCcSizesDescending();
    cout << "  Component sizes: ";
    for (int size : sizes1) {
        cout << size << " ";
    }
    cout << endl;
    
    double gamma = 0.9;
    double heuristic_return;
    double component_return;
    double auto_return;

    heuristic_return = env1.estimateRemainingReturn(0, gamma); // Heuristic
    cout << "  Heuristic rollout return: " << heuristic_return << endl;
    component_return = env1.estimateRemainingReturn(1, gamma); // Component-based
    cout << "  Component rollout return: " << component_return << endl;
    auto_return = env1.estimateRemainingReturn(-1, gamma); // Auto-select
    cout << "  Auto-selected rollout return: " << auto_return << endl;    
    
    
    // Scenario 2: Highly fragmented graph (many tiny components)
    MvcEnv env2(1.0);
    env2.s0(graph_ptr);
    // Cover more nodes to create many tiny fragments
    env2.covered_set.insert(0);  // Breaks {0,1,2,3} 
    env2.covered_set.insert(1);  
    env2.covered_set.insert(4);  // Breaks {4,5,6,7}
    env2.covered_set.insert(5);  
    
    cout << "\nScenario 2 - Highly fragmented (tiny components):" << endl;
    cout << "  Covered nodes: {0, 1, 4, 5}" << endl;
    cout << "  LCC size: " << env2.getMaxConnectedNodesNum() << endl;
    
    auto sizes2 = env2.getCcSizesDescending();
    cout << "  Component sizes: ";
    for (int size : sizes2) {
        cout << size << " ";
    }
    cout << endl;
    
    heuristic_return = env1.estimateRemainingReturn(0, gamma); // Heuristic
    cout << "  Heuristic rollout return: " << heuristic_return << endl;
    component_return = env1.estimateRemainingReturn(1, gamma); // Component-based
    cout << "  Component rollout return: " << component_return << endl;
    auto_return = env1.estimateRemainingReturn(-1, gamma); // Auto-select
    cout << "  Auto-selected rollout return: " << auto_return << endl;    
    
    return true;
}


int main(){
    test_helloworld();
    // graph function test
    test_graph();
    // env function test
    test_mvc_env();

    return 0;
}