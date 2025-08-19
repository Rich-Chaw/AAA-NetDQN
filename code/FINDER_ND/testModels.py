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
from testUtils import load_config, create_model, load_synthetic_graphs, detailed_result_dir, load_real_graph,_get_synth_param_grid

# Model configurations to compare
model_BA_params_configs = {'nrange_list':['30_50'],
                            'm_list':[1,2]}
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
    
    # Build parameter grid from eval_config['synthetic_g_params']

    synth_grid = _get_synth_param_grid(eval_config)
    for entry in synth_grid:
        config_key = entry['config_key']
        params = entry['params']
        try:
            # Load synthetic graphs (per_graphs=1)
            graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                    g_num=1,
                                    **params)

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

plot_step_ratio = 0.005

def max_cc_to_xy(MaxCCList, plot_step_ratio=0.005):
    """
    Convert MaxCCList to x,y for plotting
    Parameters:
    - MaxCCList: list of MaxCC values
    - plot_step_ratio: ratio of plot step to MaxCCList length
    Returns:
    - x: list of x values
    - y: list of y values
    """
    MaxCCList = np.array(MaxCCList)
    plot_step = max(1,np.floor(len(MaxCCList) * plot_step_ratio))
    x = np.arange(0,len(MaxCCList),plot_step).astype(int)
    if x[-1] != len(MaxCCList)-1:
        x = np.append(x, len(MaxCCList)-1)
    y = MaxCCList[x]
    x = x / (len(MaxCCList)-1)
    return x,y

def _compute_common_axes_limits(plot_step_ratio=0.05):
    """Compute common x/y limits and ticks for a set of MaxCC series.
    Returns: (x_max, x_ticks, y_max, y_ticks)
    """

    x_max = 1.0
    y_max = 1.0
    # Regularized ticks
    # X: every 0.05 up to x_max
    x_step = 0.1
    y_step = 0.1
    x_ticks = np.arange(0.0, x_max, x_step)
    y_ticks = np.arange(0.0, y_max, y_step)

    return x_max, x_ticks, y_max, y_ticks

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
            MaxCCList = np.array(results['MaxCCList'])
            x,y = max_cc_to_xy(MaxCCList)
            plt.plot(x, y, color=colors[i], linewidth=2, label=label, marker='o', markersize=3)
    
    plt.xlabel('Removal Ratio')
    plt.ylabel('Maximum Connected Component Size')
    plt.title(f'MaxCC Comparison on {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dataset_name}_max_cc_plot.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved MaxCC comparison plot: {plot_path}")
    plt.close()


def plot_synthetic_overview(all_model_results, eval_config, save_dir, model_configs):
    """Plot all synthetic config_keys in a single figure of subplots.
    Saves to synthetic_{synthetic_g_type}_max_cc.png
    """
    synth_grid = _get_synth_param_grid(eval_config)
    config_keys = [g['config_key'] for g in synth_grid]
    n = len(config_keys)
    if n == 0:
        return
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    axes = axes.flatten() if len(config_keys) > 1 else [axes]
    # Build common limits
    x_max, x_ticks, y_max, y_ticks = _compute_common_axes_limits()

    handles = []
    labels = []
    for idx, ck in enumerate(config_keys):
        ax = axes[idx]
        for model_key, model_results in all_model_results.items():
            results = model_results.get('synthetic', {}).get(ck)
            if not results or results.get('MaxCCList') is None:
                continue
            # Plot MaxCC curve
            MaxCCList = np.array(results['MaxCCList'])
            x,y = max_cc_to_xy(MaxCCList)
            model_idx = int(model_key.split('_')[1])
            mc = model_configs[model_idx]
            label = f"{mc['gnn_model']}_{mc['g_type']}_nrange_{mc['g_params']['nrange']}_m_{mc['g_params']['m']}"
            line, = ax.plot(x, y, linewidth=1.4, marker='o', markersize=2.5, label=label)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(ck, fontsize=10)
        ax.set_xlim(0, x_max if x_max > 0 else 1.0)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
    # Hide any empty axes
    for j in range(n, rows * cols):
        axes[j].axis('off')

    fig.supylabel('Maximum Connected Component Size')
    # push xlabel slightly above the bottom
    fig.supxlabel('Removal Ratio', y=0.02)
    # leave room at top for legend and at bottom for xlabel
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
    # Shared legend at the top to avoid overlapping the bottom xlabel
    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.005),
               ncol=min(len(labels), 4), fontsize=8)

    out_name = f"synthetic_{eval_config['synthetic_g_type']}_max_cc.png"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved synthetic overview plot: {out_path}")

def plot_real_overview(all_model_results, eval_config, save_dir, model_configs):
    """Plot all real datasets in a single figure of subplots and save to real_max_cc.png"""
    datasets = eval_config.get('datasets', [])
    if not datasets:
        return
    n = len(datasets)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    axes = axes.flatten() if len(datasets) > 1 else [axes]
    # Common limits
    x_max, x_ticks, y_max, y_ticks = _compute_common_axes_limits()

    handles = []
    labels = []
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for model_key, model_results in all_model_results.items():
            results = model_results.get('real', {}).get(ds)
            if not results or results.get('MaxCCList') is None:
                continue
            # Plot MaxCC curve
            MaxCCList = np.array(results['MaxCCList'])
            x,y = max_cc_to_xy(MaxCCList)
            model_idx = int(model_key.split('_')[1])
            mc = model_configs[model_idx]
            label = f"{mc['gnn_model']}_{mc['g_type']}_nrange_{mc['g_params']['nrange']}_m_{mc['g_params']['m']}"
            line, = ax.plot(x, y, linewidth=1.4, marker='o', markersize=2.5, label=label)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(ds, fontsize=10)
        ax.set_xlim(0, x_max if x_max > 0 else 1.0)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)

    # Hide any empty axes
    for j in range(n, rows * cols):
        axes[j].axis('off')

    fig.supylabel('Maximum Connected Component Size')
    fig.supxlabel('Removal Ratio', y=0.02)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.005),
               ncol=min(len(labels), 4), fontsize=8)

    out_path = os.path.join(save_dir, 'real_max_cc.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved real overview plot: {out_path}")

def save_evaluation_results(all_model_results, save_dir, eval_config):
    """
    Save evaluation results to CSV files for later analysis
    
    Parameters:
    - all_model_results: dict with all model evaluation results
    - save_dir: directory to save CSV files
    - eval_config: evaluation configuration
    """
    print(f"\nSaving evaluation results to CSV files...")
    
    # Save synthetic graph results using synthetic_g_params
    synth_grid = _get_synth_param_grid(eval_config)
    config_keys = [g['config_key'] for g in synth_grid]

    for config_key in config_keys:
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
    
    # Summaries for synthetic using synthetic_g_params

    synth_grid = _get_synth_param_grid(eval_config)
    config_keys = [g['config_key'] for g in synth_grid]

    for config_key in config_keys:
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
    
    # ---------------------Plot per-config figures (e.g., BA_nrange_30_50_m_1_max_cc_plot.png) , one graph one png file------------------------------------------
    # plot all synthetic graphs 
    synth_grid = _get_synth_param_grid(eval_config)
    config_keys = [g['config_key'] for g in synth_grid]
    for config_key in config_keys:
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
    
    #---------------------Save evaluation results to CSV--------------------------------
    # save_evaluation_results(all_model_results, save_result_dir, eval_config)

    # Print summary
    print_model_comparison_summary(all_model_results, eval_config)

    # Overview plots
    plot_synthetic_overview(all_model_results, eval_config, save_result_dir, all_model_configs)
    plot_real_overview(all_model_results, eval_config, save_result_dir, all_model_configs)

    print(f"\n{'='*60}")
    print("Model comparison evaluation completed!")
    print(f"All plots saved to: {save_result_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

    
    
