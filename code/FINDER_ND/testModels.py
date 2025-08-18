#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
from tqdm import tqdm
import numpy as np
import time
import networkx as nx
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
from testUtils import load_config, create_model, load_synthetic_graphs, detailed_result_dir, load_real_graph

# Model configurations to compare
model_BA_params_configs = {'nrange_list':['30_50'],
                            'm_list':[1,2,3,4,5,6]}
model_g_type_configs = ['BA']
model_gnn_configs = ['graphSage']

# Generate all model configurations
all_model_configs = []

for g_type in model_g_type_configs:
    for gnn in model_gnn_configs:
        if g_type == 'BA':
             for nrange in model_BA_params_configs['nrange_list']:
                for m in model_BA_params_configs['m_list']:
                    g_params = {
                        'nrange': nrange,
                        'm': m
                    }
                    model_config = {
                        "g_type": g_type,
                        "gnn_model": gnn,
                        "target_graph": "Digg",
                        "g_params": g_params,
                        "save_model_dir": "./models"
                    }
                    all_model_configs.append(model_config)
        if g_type == 'ego':
            pass

def evaluate_model_on_synthetic_graphs(dqn, model_config, eval_config):
    """
    Evaluate a model on synthetic graphs (per_graphs=1)
    
    Returns:
    - dict: results for each configuration with MaxCCList
    """
    print(f"  Evaluating on synthetic graphs...")
    
    results = {}
    
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            
            try:
                # Load synthetic graphs (per_graphs=1)
                graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                        g_num=1,  # Only 1 graph per configuration
                                        nrange=nrange, 
                                        m=m)
                
                if not graphs:
                    print(f"    Warning: No graphs found for {config_key}")
                    results[config_key] = {'score': None, 'time': None, 'MaxCCList': None}
                    continue
                
                # Evaluate on the single graph
                g = graphs[0]  # Only one graph
                temp_sol_file = f"temp_synthetic_{config_key}.txt"
                
                # Get solution and evaluate
                sol, sol_time = dqn.EvaluateRealData(g, temp_sol_file, eval_config['step_ratio'])
                score, MaxCCList = dqn.EvaluateSol(g, temp_sol_file, eval_config['strategy_id'], reInsertStep=0.001)
                
                # Clean up temp file
                if os.path.exists(temp_sol_file):
                    os.remove(temp_sol_file)
                
                results[config_key] = {
                    'score': score,
                    'time': sol_time,
                    'MaxCCList': MaxCCList,
                    'graph': g
                }
                
                print(f"    ✓ {config_key}: score={score:.6f}, time={sol_time:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Error evaluating {config_key}: {e}")
                results[config_key] = {'score': None, 'time': None, 'MaxCCList': None}
    
    return results

def evaluate_model_on_real_graphs(dqn, model_config, eval_config, data_config):
    """
    Evaluate a model on real graphs
    
    Returns:
    - dict: results for each dataset with MaxCCList
    """
    print(f"  Evaluating on real graphs...")
    
    results = {}
    datasets = eval_config['datasets']
    
    for dataset in datasets:
        try:
            # Load real graph
            dataset_dir = os.path.join(data_config['dataset_dir'], "real")
            g = load_real_graph(dataset, dataset_dir)
            
            if g is None:
                print(f"    Warning: Could not load graph for {dataset}")
                results[dataset] = {'score': None, 'time': None, 'MaxCCList': None}
                continue
            
            # Create temp solution file
            temp_sol_file = f"temp_real_{dataset}.txt"
            
            # Get solution and evaluate
            sol, sol_time = dqn.EvaluateRealData(g, temp_sol_file, eval_config['step_ratio'])
            score, MaxCCList = dqn.EvaluateSol(g, temp_sol_file, eval_config['strategy_id'], reInsertStep=0.001)
            
            # Clean up temp file
            if os.path.exists(temp_sol_file):
                os.remove(temp_sol_file)
            
            results[dataset] = {
                'score': score,
                'time': sol_time,
                'MaxCCList': MaxCCList,
                'graph': g
            }
            
            print(f"    ✓ {dataset}: score={score:.6f}, time={sol_time:.2f}s")
            
        except Exception as e:
            print(f"    ✗ Error evaluating {dataset}: {e}")
            results[dataset] = {'score': None, 'time': None, 'MaxCCList': None}
    
    return results

def plot_max_cc_comparison(all_results, dataset_name, save_dir, model_configs):
    """
    Plot MaxCC comparison for different models on the same dataset/graph
    
    Parameters:
    - all_results: dict with model_key -> results mapping
    - dataset_name: name of the dataset or configuration
    - save_dir: directory to save the plot
    - model_configs: list of model configurations for labels
    """
    plt.figure(figsize=(10, 6))
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for i, (model_key, results) in enumerate(all_results.items()):
        if results['MaxCCList'] is not None:
            # Extract model index from model_key (e.g., "model_0" -> 0)
            model_idx = int(model_key.split('_')[1])
            model_config = model_configs[model_idx]
            
            # Create model label
            g_type = model_config['g_type']
            gnn = model_config['gnn_model']
            if g_type == 'BA':
                nrange = model_config['g_params']['nrange']
                m = model_config['g_params']['m']
                label = f"{gnn}_{g_type}_nrange_{nrange}_m_{m}"
            else:
                label = f"{gnn}_{g_type}"
            
            # Plot MaxCC curve
            MaxCCList = results['MaxCCList']
            # Use step_ratio from config or default to 0.01
            step_ratio = 0.01  # Default value, you can modify this if needed
            x = np.arange(len(MaxCCList)) * step_ratio
            plt.plot(x, MaxCCList, color=colors[i], linewidth=2, label=label, marker='o', markersize=3)
    
    plt.xlabel('Removal Ratio')
    plt.ylabel('Maximum Connected Component Size')
    plt.title(f'MaxCC Comparison on {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dataset_name}_max_cc.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved MaxCC comparison plot: {plot_path}")
    plt.close()

def save_evaluation_results(all_model_results, save_dir, eval_config):
    """
    Save evaluation results to CSV files for later analysis
    
    Parameters:
    - all_model_results: dict with all model evaluation results
    - save_dir: directory to save CSV files
    - eval_config: evaluation configuration
    """
    print(f"\nSaving evaluation results to CSV files...")
    
    # Save synthetic graph results
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            csv_filename = f"synthetic_{config_key}_model_comparison.csv"
            csv_path = os.path.join(save_dir, csv_filename)
            
            # Prepare data for CSV
            csv_data = []
            for model_key, model_results in all_model_results.items():
                if (model_results['synthetic'] and 
                    config_key in model_results['synthetic'] and 
                    model_results['synthetic'][config_key]['score'] is not None):
                    
                    result = model_results['synthetic'][config_key]
                    model_idx = int(model_key.split('_')[1])
                    model_config = model_results['config']
                    
                    csv_data.append({
                        'model_index': model_idx,
                        'g_type': model_config['g_type'],
                        'gnn_model': model_config['gnn_model'],
                        'nrange': model_config['g_params']['nrange'],
                        'm': model_config['g_params']['m'],
                        'best_iter': model_results['best_iter'],
                        'score': result['score'],
                        'time': result['time'],
                        'max_cc_list_length': len(result['MaxCCList']) if result['MaxCCList'] else 0
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                print(f"  ✓ Saved synthetic results: {csv_filename} ({len(df)} models)")
    
    # Save real graph results
    for dataset in eval_config['datasets']:
        csv_filename = f"real_{dataset}_model_comparison.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        
        # Prepare data for CSV
        csv_data = []
        for model_key, model_results in all_model_results.items():
            if (model_results['real'] and 
                dataset in model_results['real'] and 
                model_results['real'][dataset]['score'] is not None):
                
                result = model_results['real'][dataset]
                model_idx = int(model_key.split('_')[1])
                model_config = model_results['config']
                
                csv_data.append({
                    'model_index': model_idx,
                    'g_type': model_config['g_type'],
                    'gnn_model': model_config['gnn_model'],
                    'nrange': model_config['g_params']['nrange'],
                    'm': model_config['g_params']['m'],
                    'best_iter': model_results['best_iter'],
                    'score': result['score'],
                    'time': result['time'],
                    'max_cc_list_length': len(result['MaxCCList']) if result['MaxCCList'] else 0
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved real graph results: {csv_filename} ({len(df)} models)")
    
    print(f"  ✓ All evaluation results saved to CSV files")

def print_model_comparison_summary(all_model_results, eval_config):
    """
    Print a summary of model comparison results
    
    Parameters:
    - all_model_results: dict with all model evaluation results
    - eval_config: evaluation configuration
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Summary for synthetic graphs
    print(f"\nSynthetic Graph Evaluation Results:")
    print(f"{'='*40}")
    
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            print(f"\n{config_key}:")
            
            # Collect scores for this configuration
            scores = []
            for model_key, model_results in all_model_results.items():
                if (model_results['synthetic'] and 
                    config_key in model_results['synthetic'] and 
                    model_results['synthetic'][config_key]['score'] is not None):
                    
                    score = model_results['synthetic'][config_key]['score']
                    model_idx = int(model_key.split('_')[1])
                    model_config = model_results['config']
                    scores.append((score, model_idx, model_config))
            
            if scores:
                # Sort by score (lower is better for dismantling)
                scores.sort(key=lambda x: x[0])
                print(f"  Best model: Model {scores[0][1]} (score: {scores[0][0]:.6f})")
                print(f"  Worst model: Model {scores[-1][1]} (score: {scores[-1][0]:.6f})")
                print(f"  Score range: {scores[-1][0] - scores[0][0]:.6f}")
            else:
                print(f"  No valid results")
    
    # Summary for real graphs
    print(f"\nReal Graph Evaluation Results:")
    print(f"{'='*40}")
    
    for dataset in eval_config['datasets']:
        print(f"\n{dataset}:")
        
        # Collect scores for this dataset
        scores = []
        for model_key, model_results in all_model_results.items():
            if (model_results['real'] and 
                dataset in model_results['real'] and 
                model_results['real'][dataset]['score'] is not None):
                
                score = model_results['real'][dataset]['score']
                model_idx = int(model_key.split('_')[1])
                model_config = model_results['config']
                scores.append((score, model_idx, model_config))
        
        if scores:
            # Sort by score (lower is better for dismantling)
            scores.sort(key=lambda x: x[0])
            print(f"  Best model: Model {scores[0][1]} (score: {scores[0][0]:.6f})")
            print(f"  Worst model: Model {scores[-1][1]} (score: {scores[-1][0]:.6f})")
            print(f"  Score range: {scores[-1][0] - scores[0][0]:.6f}")
        else:
            print(f"  No valid results")
    
    print(f"\n{'='*60}")

def main():
    # Load configuration
    config = load_config()
    eval_config = config['eval_config']
    data_config = config['data_config']
    
    # Set per_graphs to 1 for synthetic evaluation
    eval_config['per_graphs'] = 1
    
    print(f"Model Comparison Evaluation")
    print(f"Number of models to compare: {len(all_model_configs)}")
    print(f"Synthetic graphs per config: {eval_config['per_graphs']}")
    print(f"Real datasets: {eval_config['datasets']}")
    
    # Create save directory
    save_result_dir = eval_config['save_result_dir']
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {save_result_dir}")
    
    # Store all results for plotting
    all_model_results = {}
    
    # Evaluate each model
    for i, model_config in enumerate(tqdm(all_model_configs, desc="Evaluating models")):
        print(f"\n{'='*60}")
        print(f"Model {i+1}/{len(all_model_configs)}: {model_config['g_type']}_nrange_{model_config['g_params']['nrange']}_m_{model_config['g_params']['m']}")
        print(f"{'='*60}")
        
        try:
            # Create and load model
            dqn = create_model(model_config)
            best_ckpt_file = dqn.findModel()
            best_iter = int(best_ckpt_file.split('.ckpt')[0].split('_')[-1])
            dqn.LoadModel(best_ckpt_file)
            
            print(f"  ✓ Loaded model: {best_ckpt_file} (iteration {best_iter})")
            
            # Evaluate on synthetic graphs
            synthetic_results = evaluate_model_on_synthetic_graphs(dqn, model_config, eval_config)
            
            # Evaluate on real graphs
            real_results = evaluate_model_on_real_graphs(dqn, model_config, eval_config, data_config)
            
            # Store results
            model_key = f"model_{i}"
            all_model_results[model_key] = {
                'synthetic': synthetic_results,
                'real': real_results,
                'config': model_config,
                'best_iter': best_iter
            }
            
        except Exception as e:
            print(f"  ✗ Error evaluating model {i+1}: {e}")
            continue
    
    # Create comparison plots
    print(f"\n{'='*60}")
    print("Creating MaxCC comparison plots...")
    print(f"{'='*60}")
    
    # Plot synthetic graph comparisons
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            
            # Collect results for this configuration across all models
            synthetic_results = {}
            for model_key, model_results in all_model_results.items():
                if model_results['synthetic'] and config_key in model_results['synthetic']:
                    synthetic_results[model_key] = model_results['synthetic'][config_key]
            
            if synthetic_results:
                print(f"\nPlotting MaxCC comparison for {config_key}...")
                plot_max_cc_comparison(synthetic_results, config_key, save_result_dir, all_model_configs)
    
    # Plot real graph comparisons
    for dataset in eval_config['datasets']:
        # Collect results for this dataset across all models
        real_results = {}
        for model_key, model_results in all_model_results.items():
            if model_results['real'] and dataset in model_results['real']:
                real_results[model_key] = model_results['real'][dataset]
        
        if real_results:
            print(f"\nPlotting MaxCC comparison for {dataset}...")
            plot_max_cc_comparison(real_results, dataset, save_result_dir, all_model_configs)
    
    # Save evaluation results to CSV
    save_evaluation_results(all_model_results, save_result_dir, eval_config)

    # Print summary
    print_model_comparison_summary(all_model_results, eval_config)

    print(f"\n{'='*60}")
    print("Model comparison evaluation completed!")
    print(f"All plots saved to: {save_result_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

    
    
