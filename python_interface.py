#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Python interface for AAA-NetDQN network dismantling
Usage: python python_interface.py --graph_file <path_to_graph> [--model_path <path>] [--step_ratio <ratio>]
"""

import sys
import os
import argparse
import json
import pickle
import tempfile
import networkx as nx
import igraph as ig
import numpy as np
import pathlib

# Disable CUDA to avoid GPU issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_model_and_config(model_path=None):
    """Load the DQN model and configuration"""
    # Add code directory to path
    code_dir = os.path.join(os.path.dirname(__file__), 'code')
    sys.path.append(code_dir)
    # Add FINDER directory to path for GraphDQN
    finder_dir = os.path.join(code_dir, 'FINDER')
    sys.path.append(finder_dir)
    
    try:
        # Import required modules - GraphDQN is a compiled module
        from testUtils import load_config, create_model
        
        # Try to import GraphDQN (compiled module)
        try:
            from GraphDQN import GraphDQN
            DQN_cls = GraphDQN
        except ImportError:
            # Fallback: try to import from the main code directory
            sys.path.insert(0, code_dir)
            import GraphDQN as GraphDQN_module
            DQN_cls = GraphDQN_module.GraphDQN
        
        # FINDER_type = args.model_path.split("/")[1]
        # # sys.path.append(os.path.dirname(__file__) + os.sep + '../')
        # if "moe" in FINDER_type:
        #     sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        #     sys.path.append(os.path.dirname(__file__) + os.sep + f'FINDER/')
        #     from GraphDQN import GraphDQN
        #     from MoEGraphDQN import MoEGraphDQN
        # elif "advance" in FINDER_type:
        #     sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        #     from AdvanceGraphDQN import AdvanceGraphDQN
        #     DQN_cls = AdvanceGraphDQN
        # else:
        #     sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        #     from GraphDQN import GraphDQN
        #     DQN_cls = GraphDQN


        # Load configuration
        config_file = os.path.join(model_path,"config.json")
        config = load_config(config_file)
        
        model_config = config['model_config']
        model_config['save_model_dir'] = os.path.dirname(model_path)
        # Load best model
        if "SOTA" in model_path:
            # original released FINDER model
            dqn = create_model(model_config, DQN_cls,iter=78000)
        else:
            dqn = create_model(model_config, DQN_cls)
            best_ckpt = dqn.findModel()
            dqn.LoadModel(best_ckpt)
        
        return dqn, config
        
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        raise

def dismantle_graph(graph, model_path=None, step_ratio=0.01, strategy_id=0, reInsertStep=0.001):
    """
    Dismantle a NetworkX graph using the AAA-NetDQN model
    
    Args:
        graph: NetworkX graph
        model_path: Path to model directory (optional)
        step_ratio: Step ratio for dismantling (default: 0.01)
        strategy_id: Strategy ID for evaluation (default: 0)
    """
    try:
        # Load model and config
        dqn, config = load_model_and_config(model_path)
        
        # Create temporary solution file
        temp_sol_fd, temp_sol_file = tempfile.mkstemp(suffix='_sol.txt')
        os.close(temp_sol_fd)
        
        try:
            # Load graph in the format expected by the model
            g_test = graph.copy()
            sol, sol_time = dqn.EvaluateRealData(g_test, temp_sol_file, step_ratio)
            score, MaxCCList, solution = dqn.EvaluateSol(g_test, temp_sol_file, strategy_id, reInsertStep=reInsertStep,log_removals=True)
            
            return solution, float(score), MaxCCList
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_sol_file):
                os.remove(temp_sol_file)
                
    except Exception as e:
        print(f"Error in dismantle_graph: {e}", file=sys.stderr)
        return [], 0.0

def main():
    parser = argparse.ArgumentParser(description='Network dismantling interface')
    parser.add_argument('--graph_file', type=str, help='Path to graph file (pickle or edgelist)')
    parser.add_argument('--model_path', type=str,default='./code/FINDER/models/graphSage_BA_nrange_70_90_m_5', help='Path to model directory')
    parser.add_argument('--step_ratio', type=float, default=0.01, help='Step ratio for dismantling')
    parser.add_argument('--strategy_id', type=int, default=0, help='Strategy ID for evaluation')
    parser.add_argument('--out_file',type=str, help='Output path')
    
    args = parser.parse_args()
    
    # args.graph_file = '../dataset/real/Crime.txt'
    # args.model_path = './code/FINDER/models/graphSage_BA_nrange_30_50_m_4_SOTA'

    # Load graph
    graph = None
    if args.graph_file:
        suffix = pathlib.Path(args.graph_file).suffix
        if suffix == '.pkl': #igraph or networkx
            with open(args.graph_file, 'rb') as f:
                # Load the pickled graph
                graph = pickle.load(f)
                
                # Check if it's an igraph and convert to NetworkX
                if isinstance(graph, ig.Graph):
                    edgelist = graph.get_edgelist()
                    graph = nx.Graph()
                    graph.add_edges_from(edgelist)

                elif isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                    pass
                else:
                    print(f"Warning: Unknown graph type: {type(graph)}", file=sys.stderr)
                
        elif suffix == '.txt':
            graph = nx.read_edgelist(args.graph_file, nodetype=int)
        elif suffix == '.gml':
            graph = nx.read_gml(args.graph_file)
    if graph is None:
        print("Error: Failed to load graph", file=sys.stderr)
        return 1
    
    # Dismantle graph
    removals, score, MaxCCList = dismantle_graph(
        graph, 
        model_path=args.model_path,
        step_ratio=args.step_ratio,
        strategy_id=args.strategy_id
    )

    # Output results
    if args.out_file:
        result = {
            'removals': removals,
            'robustness': score,
            'MaxCCList': MaxCCList,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }
        with open(args.out_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())