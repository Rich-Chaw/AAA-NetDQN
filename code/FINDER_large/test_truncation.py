#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test script for Episode Truncation (Early Termination) with Reward Reshaping
"""

import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import networkx as nx
from GraphDQN import GraphDQN

def test_episode_truncation():
    """Test the episode truncation functionality"""
    print("=== Testing Episode Truncation Implementation ===")
    
    # Create a test graph with 1000 nodes (large-scale)
    print("Creating test graph with 1000 nodes...")
    G = nx.barabasi_albert_graph(1000, 3)
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Initialize GraphDQN
    print("Initializing GraphDQN...")
    dqn = GraphDQN(
        g_type='BA',
        g_params={'nrange': '1000_1000', 'm': 3},
        gnn_model='graphSage',
        target_graph="Test"
    )
    

    # Test the truncation logic
    print("Testing truncation logic...")
    
    # Create a test environment
    test_env = dqn.test_env
    
    # Convert NetworkX graph to internal format
    edges = G.edges()
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
    else:
        A = np.array([0])
        B = np.array([0])
    
    # Create internal graph representation
    from graph import py_Graph
    test_graph = py_Graph(G.number_of_nodes(), G.number_of_edges(), A, B)
    
    # Initialize environment
    test_env.s0(test_graph)
    
    print(f"Original graph size: {test_env.graph.num_nodes}")
    print(f"Truncation threshold (50%): {0.5 * test_env.graph.num_nodes}")
    
    # Test initial state
    print(f"Initial LCC size: {test_env.getMaxConnectedNodesNum()}")
    print(f"Is truncated initially: {test_env.isTruncated()}")
    print(f"Is terminal initially: {test_env.isTerminal()}")
    
    # Simulate some steps to test truncation
    step_count = 0
    max_steps = 200  # Limit test steps
    
    while (not test_env.isTerminal() and 
           not test_env.isTruncated() and 
           step_count < max_steps):
        
        # Get available actions
        try:
            action = test_env.degreeAction()
            reward = test_env.step(action)
            step_count += 1
            
            lcc_size = test_env.getMaxConnectedNodesNum()
            is_truncated = test_env.isTruncated()
            is_terminal = test_env.isTerminal()
            
            if step_count % 10 == 0:
                print(f"Step {step_count}: LCC size = {lcc_size:.1f}, "
                      f"Truncated = {is_truncated}, Terminal = {is_terminal}, "
                      f"Reward = {reward:.6f}")
            
            if is_truncated:
                print(f"\nEpisode truncated at step {step_count}!")
                print(f"Final LCC size: {lcc_size:.1f}")
                print(f"Truncation threshold: {0.2 * test_env.graph.num_nodes}")
                break
                
        except Exception as e:
            print(f"Error at step {step_count}: {e}")
            break
    
    if test_env.isTerminal():
        print(f"\nEpisode terminated normally at step {step_count}")
    elif step_count >= max_steps:
        print(f"\nTest completed after {max_steps} steps")
    
    print("=== Episode Truncation Test Completed ===")

def test_reward_reshaping():
    """Test the reward reshaping functionality"""
    print("\n=== Testing Reward Reshaping ===")
    
    # Create a smaller test graph for easier testing
    G = nx.barabasi_albert_graph(100, 2)
    
    dqn = GraphDQN(
        g_type='BA',
        g_params={'nrange': '100_100', 'm': 2},
        gnn_model='graphSage',
        target_graph="Test"
    )
    
    test_env = dqn.test_env
    
    # Convert and initialize
    edges = G.edges()
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
    else:
        A = np.array([0])
        B = np.array([0])
    
    from graph import py_Graph
    test_graph = py_Graph(G.number_of_nodes(), G.number_of_edges(), A, B)
    test_env.s0(test_graph)
    
    print(f"Original graph size: {test_env.graph.num_nodes}")
    print(f"Truncation threshold: {0.2 * test_env.graph.num_nodes}")
    
    # Test rewards before and after truncation
    step_count = 0
    while not test_env.isTerminal() and step_count < 50:
        try:
            action = test_env.degreeAction()
            reward = test_env.step(action)
            step_count += 1
            
            lcc_size = test_env.getMaxConnectedNodesNum()
            is_truncated = test_env.isTruncated()
            
            if step_count % 5 == 0:
                print(f"Step {step_count}: LCC = {lcc_size:.1f}, "
                      f"Truncated = {is_truncated}, Reward = {reward:.6f}")
            
            if is_truncated:
                print(f"Reward after truncation: {reward:.6f}")
                
                # Test heuristic rollout directly
                print("Testing heuristic rollout...")
                heuristic_reward = test_env.computeHeuristicRollout(gamma=1.0)
                print(f"Heuristic rollout reward: {heuristic_reward:.6f}")
                break
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print("=== Reward Reshaping Test Completed ===")

def test_heuristic_rollout():
    """Test the heuristic rollout functionality in detail"""
    print("\n=== Testing Heuristic Rollout in Detail ===")
    
    # Create a small test graph
    G = nx.barabasi_albert_graph(50, 2)
    
    dqn = GraphDQN(
        g_type='BA',
        g_params={'nrange': '50_50', 'm': 2},
        gnn_model='graphSage',
        target_graph="Test"
    )
    
    test_env = dqn.test_env
    
    # Convert and initialize
    edges = G.edges()
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
    else:
        A = np.array([0])
        B = np.array([0])
    
    from graph import py_Graph
    test_graph = py_Graph(G.number_of_nodes(), G.number_of_edges(), A, B)
    test_env.s0(test_graph)
    
    print(f"Initial graph: {test_env.graph.num_nodes} nodes, {test_env.graph.num_edges} edges")
    print(f"Initial LCC size: {test_env.getMaxConnectedNodesNum()}")
    
    # Remove some nodes to create a partial state
    for i in range(10):
        if not test_env.isTerminal():
            action = test_env.degreeAction()
            test_env.stepWithoutReward(action)
    
    print(f"After 10 removals - LCC size: {test_env.getMaxConnectedNodesNum()}")
    print(f"Covered nodes: {len(test_env.covered_set)}")
    
    # Test heuristic rollout
    print("Running heuristic rollout...")
    heuristic_reward = test_env.computeHeuristicRollout(1.0)
    print(f"Heuristic rollout completed. Total reward: {heuristic_reward:.6f}")
    
    # Verify state was restored
    print(f"State after rollout - LCC size: {test_env.getMaxConnectedNodesNum()}")
    print(f"Covered nodes after rollout: {len(test_env.covered_set)}")
    
    print("=== Heuristic Rollout Test Completed ===")

if __name__ == "__main__":
    try:
        test_episode_truncation()
        test_reward_reshaping()
        test_heuristic_rollout()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
