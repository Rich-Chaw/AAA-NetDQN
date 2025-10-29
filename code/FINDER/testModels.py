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
from testUtils import load_real_csv,load_synth_csv


def _group_models_by_params(model_configs):
    """Groups model configurations by g_type and nrange for better visualization.

    Returns:
    - dict: A dictionary where keys are group identifiers (e.g., 'BA_nrange_30_50')
            and values are lists of model configurations belonging to that group.
    """
    grouped_models = {}
    for model_config in model_configs:
        g_type = model_config['g_type']
        nrange = model_config['g_params']['nrange']
        group_key = f'{g_type}_nrange_{nrange}'

        if group_key not in grouped_models:
            grouped_models[group_key] = []
        grouped_models[group_key].append(model_config)
    return grouped_models

def evaluate_model_on_synthetic_graphs(dqn, model_config, eval_config):
    """
    Evaluate a model on synthetic graphs (per_graphs=1)
    
    Returns:
    - results : dict 
    {dataset_name:{
                'score': score,
                'time': sol_time,
                'MaxCCList': MaxCCList,
                'graph': g
            }}
    """
    print(f"  Evaluating on synthetic graphs...")
    
    results = {}
    
    # Build parameter grid from eval_config['synthetic_g_params']

    synth_grid = _get_synth_param_grid(eval_config)
    for entry in synth_grid:
        # synth_dataset: '{g_type}_nrange_{nrange}_m_{m}'
        synth_dataset = entry['synth_dataset']
        params = entry['params']
        try:
            # Load synthetic graphs (per_graphs=1)
            graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                    g_num=1,
                                    **params)

            if not graphs:
                print(f"    Warning: No graphs found for {synth_dataset}")
                results[synth_dataset] = {'score': None, 'time': None, 'MaxCCList': None}
                continue
            
            # Evaluate on the single graph
            g = graphs[0]  # Only one graph
            temp_sol_file = f"temp_synthetic_{synth_dataset}.txt"
            
            # Get solution and evaluate
            sol, sol_time = dqn.EvaluateRealData(g, temp_sol_file, eval_config['step_ratio'])
            score, MaxCCList = dqn.EvaluateSol(g, temp_sol_file, eval_config['strategy_id'], reInsertStep=0.001)
            
            # Clean up temp file
            if os.path.exists(temp_sol_file):
                os.remove(temp_sol_file)
            
            results[synth_dataset] = {
                'score': score,
                'time': sol_time,
                'MaxCCList': MaxCCList,
                'graph': g
            }
            
            print(f"    ✓ {synth_dataset}: score={score:.6f}, time={sol_time:.2f}s")
                
        except Exception as e:
            print(f"    ✗ Error evaluating {synth_dataset}: {e}")
            results[synth_dataset] = {'score': None, 'time': None, 'MaxCCList': None}
    
    return results

def evaluate_model_on_real_graphs(dqn, model_config, eval_config, data_config):
    """
    Evaluate a model on real graphs
    
    Returns:
    - results : dict
    {dataset_name:{
                'score': score,
                'time': sol_time,
                'MaxCCList': MaxCCList,
                'graph': g
            }}
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


def plot_max_cc_comparison(all_results, save_dir, model_configs):
    """
    Plot MaxCC comparison for different models on the same dataset/graph
    
    Parameters:
    - all_results: dict {dataset:{model_key:model_results}}
    - dataset: name of the dataset or configuration
    - save_dir: directory to save the plot
    - model_configs: list of model configurations for labels
    """
    
    for dataset,dataset_results in all_results.items():
        plt.figure(figsize=(10, 6))
        print(f"Plotting MaxCC comparison on {dataset}")
        
        # Colors for different models
        colors = plt.cm.tab20(np.linspace(0, 1, len(dataset_results)))

        for i, (model_key, results) in enumerate(dataset_results.items()):
            if results['MaxCCList'] is not None:
                # Extract model index from model_key (e.g., "model_1" -> 1)
                model_idx = int(model_key.split('_')[1])
                model_config = model_configs[model_idx-1]
                
                # Create model label
                g_type = model_config['g_type']
                gnn = model_config['gnn_model']
                if g_type == 'BA':
                    nrange = model_config['g_params']['nrange']
                    m = model_config['g_params']['m']
                    label = f"model_{gnn}_{g_type}_nrange_{nrange}_m_{m}"
                else:
                    label = f"model_{gnn}_{g_type}"
                
                # Plot MaxCC curve
                MaxCCList = np.array(results['MaxCCList'])
                x,y = max_cc_to_xy(MaxCCList)
                plt.plot(x, y, color=colors[i], linewidth=2, label=label, marker='o', markersize=3)
    
        plt.xlabel('Removal Ratio')
        plt.ylabel('Maximum Connected Component Size')
        plt.title(f'Dataset : {dataset}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{dataset}_max_cc_plot.png"
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved MaxCC comparison plot: {plot_path}")
        plt.close()


def plot_best_iter_overview(all_model_results, eval_config, save_dir, model_configs, mode='synthetic'):
    """Plot all datasets in a single figure of subplots, grouped by model parameters.
    Saves to {mode}_grouped_max_cc.png
    """
    if mode == 'synthetic':
        synth_grid = _get_synth_param_grid(eval_config)
        datasets = [g['synth_dataset'] for g in synth_grid]
        g_type_for_filename = 'synthetic_'+ eval_config['synthetic_g_type']
    elif mode == 'real':
        datasets = eval_config.get('datasets', [])
        g_type_for_filename = 'real'
    else:
        print(f"  ⚠ Invalid mode '{mode}' for plot_best_iter_overview.")
        return
    
    # row and col
    n = len(datasets)
    m = 0
    if n == 0:
        print(f"  ⚠ No {mode} datasets found to plot for overview.")
        return
    
    grouped_models = _group_models_by_params(model_configs)
    
    all_plot_combinations = []
    for dataset in datasets:
        m = max(m,len(grouped_models.keys()))
        for group_key in grouped_models.keys():
            all_plot_combinations.append((dataset, group_key))
    
    n_plots = n*m
    if n_plots == 0:
        print(f"  ⚠ No {mode} iteration results found to plot for overview after grouping.")
        return
    rows = n
    cols = m

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*']
    # Build common limits
    x_max, x_ticks, y_max, y_ticks = _compute_common_axes_limits()

    for plot_idx, (dataset, group_key) in enumerate(all_plot_combinations):
        ax = axes[plot_idx]

        group_model_configs = grouped_models[group_key]
        local_handles = []
        local_labels = []
        fixed_line_style_for_group = '-' # All models within a group share solid line style
        
        colors_for_models_in_group = plt.cm.get_cmap('tab10', len(group_model_configs))

        for m_idx_in_group, model_config in enumerate(group_model_configs):
            model_key = next((mk for mk, mr in all_model_results.items() if mr['config'] == model_config), None)
            if not model_key:
                continue
            
            results = all_model_results.get(model_key, {}).get(mode, {}).get(dataset)
            if not results or results.get('MaxCCList') is None:
                continue
            
            MaxCCList = np.array(results['MaxCCList'])
            x,y = max_cc_to_xy(MaxCCList)
            
            m_value = model_config['g_params']['m']
            
            marker_style = markers[m_idx_in_group % len(markers)]
            model_specific_color = colors_for_models_in_group(m_idx_in_group)
            
            label = f"m_{m_value}"
            
            line, = ax.plot(x, y, linewidth=1.4, marker=marker_style, markersize=2.5,
                            linestyle=fixed_line_style_for_group, color=model_specific_color, label=label)
            
            if label not in local_labels:
                local_handles.append(line)
                local_labels.append(label)
        
        ax.set_title(f'{dataset} - {group_key}', fontsize=10) 
        ax.set_xlim(0, x_max if x_max > 0 else 1.0)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        if local_handles:
            ax.legend(handles=local_handles, labels=local_labels, loc='upper right', fontsize=8)

    # Hide any empty axes
    for j in range(n_plots, rows * cols):
        axes[j].axis('off')

    fig.supylabel('Maximum Connected Component Size')
    fig.supxlabel('Removal Ratio', y=0.02)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])

    out_name = f"{g_type_for_filename}_overview_max_cc.png"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved MaxCC best_iter overview plot: {out_path}")
    plt.close(fig)


def plot_all_iters_overview(all_model_results, eval_config, save_dir, model_configs,mode='real'):
    """Plot evaluation scores for all iterations (from iters_synthetic and iters_real) in a single figure of subplots.
    Saves to iters_overview_max_cc.png
    """
    # Collect all unique datasets from iters_synthetic and iters_real across all models
    datasets = set()
    for model_key, model_results in all_model_results.items():
        if mode == 'synthetic' and 'iters_synthetic' in model_results:
            datasets.update(model_results['iters_synthetic'].keys())
        if  mode == 'real' and 'iters_real' in model_results:
            datasets.update(model_results['iters_real'].keys())
    
    datasets = sorted(list(datasets))
    n = len(datasets)
    m = 0
    if n == 0:
        print("  ⚠ No iteration results found to plot for overview.")
        return

    # rows = int(np.ceil(np.sqrt(n)))
    # cols = int(np.ceil(n / rows))
    # fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    # axes = axes.flatten() if rows*cols > 1 else [axes]

    grouped_models = _group_models_by_params(model_configs)

    all_plot_combinations = []
    for dataset in datasets:
        m = max(m,len(grouped_models.keys()))
        for group_key in grouped_models.keys():
            all_plot_combinations.append((dataset, group_key))
    
    n_plots = n*m
    if n_plots == 0:
        print("  ⚠ No iteration results found to plot for overview after grouping.")
        return
    rows = n
    cols = m
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*']
    # Common limits (using iteration range from the data)
    max_iter_val = 0
    min_iter_val = float('inf')
    for model_key, model_results in all_model_results.items():
        for dataset_type in ['iters_synthetic', 'iters_real']:
            if dataset_type in model_results and model_results[dataset_type]:
                for dataset, iters_data in model_results[dataset_type].items():
                    if iters_data:
                        iters = np.array(list(iters_data.keys()), dtype=int)
                        if len(iters) > 0:
                            max_iter_val = max(max_iter_val, np.max(iters))
                            min_iter_val = min(min_iter_val, np.min(iters))
    
    x_max = max_iter_val + 500 if max_iter_val > 0 else 6000 # Add some buffer
    y_max = 0.55 # Score is usually 0-1
    x_step_dynamic = max(500, int(x_max / 5)) # Dynamic step for iterations
    x_ticks = np.arange(0, x_max + x_step_dynamic, x_step_dynamic)
    y_ticks = np.arange(0.0, y_max + 0.1, 0.1)

    for plot_idx, (dataset, group_key) in enumerate(all_plot_combinations):
        ax = axes[plot_idx]

        group_model_configs = grouped_models[group_key]
        local_handles = []
        local_labels = []
        fixed_line_style_for_group = '-' # All models within a group share solid line style

        colors_for_models_in_group = plt.cm.get_cmap('tab10', len(group_model_configs))

        for m_idx_in_group, model_config in enumerate(group_model_configs):
            model_key = next((mk for mk, mr in all_model_results.items() if mr['config'] == model_config), None)
            if not model_key:
                continue
            
            current_iter_results = None
            if 'iters_synthetic' in all_model_results[model_key] and dataset in all_model_results[model_key]['iters_synthetic']:
                current_iter_results = all_model_results[model_key]['iters_synthetic'][dataset]
            elif 'iters_real' in all_model_results[model_key] and dataset in all_model_results[model_key]['iters_real']:
                current_iter_results = all_model_results[model_key]['iters_real'][dataset]

            if not current_iter_results:
                continue
            
            iters = []
            scores = []
            for iter_num in sorted(current_iter_results.keys()):
                if current_iter_results[iter_num] and current_iter_results[iter_num].get('score') is not None:
                    iters.append(iter_num)
                    scores.append(current_iter_results[iter_num]['score'])
            
            if not iters:
                continue
            
            m_value = model_config['g_params']['m']
            
            marker_style = markers[m_idx_in_group % len(markers)]
            model_specific_color = colors_for_models_in_group(m_idx_in_group)
            
            label = f"m_{m_value}"
            
            line, = ax.plot(iters, scores, linewidth=1.4, marker=marker_style, markersize=2.5,
                            linestyle=fixed_line_style_for_group, color=model_specific_color, label=label)
            
            if label not in local_labels:
                local_handles.append(line)
                local_labels.append(label)
        
        ax.set_title(f'Scores on {dataset} - {group_key}', fontsize=10)
        ax.set_xlim(min_iter_val - 100 if min_iter_val != float('inf') else 0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        if local_handles:
            ax.legend(handles=local_handles, labels=local_labels, loc='upper right', fontsize=8)

    # Hide any empty axes
    # for j in range(n_plots, rows * cols):
    #     axes[j].axis('off')

    fig.supylabel('Solution Score')
    fig.supxlabel('Iteration', y=0.02)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
    # fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.005),
    #            ncol=min(len(labels), 4), fontsize=8)

    out_path = os.path.join(save_dir, f'iters_{mode}_overview_max_cc.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved MaxCC all_iters overview plot: {out_path}")
    plt.close(fig)


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
    synth_datasets = [g['synth_dataset'] for g in synth_grid]

    for synth_dataset in synth_datasets:
        print(f"\n{synth_dataset}:")
            
        # Collect scores for this configuration
        scores = []
        for model_key, model_results in all_model_results.items():
            if (model_results['synthetic'] and 
                synth_dataset in model_results['synthetic'] and 
                model_results['synthetic'][synth_dataset]['score'] is not None):
                
                score = model_results['synthetic'][synth_dataset]['score']
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
    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']

    
    parser = argparse.ArgumentParser(description='Evaluate all models on synthetic and real datasets')
    parser.add_argument("--model_path", type=str, help="Path to model directory (e.g., ./models/BA_nrange_30_50_m_3)")
    parser.add_argument("--min_iter", type=int, default=0, help="Minimum iteration to evaluate")
    parser.add_argument("--max_iter", type=int, default=6000, help="Maximum iteration to evaluate")
    parser.add_argument("--iter_step", type=int, default=300, help="Iteration step size")
    parser.add_argument("--per_graphs", type=int, default=2, help="Number of graphs per configuration")
    parser.add_argument("--eval_iter",type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--eval_real", action="store_true")
    parser.add_argument("--eval_synthetic", action="store_true")
    parser.add_argument("--eval_all_iters", action="store_true", help="plot all iter results in csv from result dir")
    args = parser.parse_args()

    # Update model config to compare
    model_BA_params_configs = {'nrange_list':['30_50','50_100'],
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
                        # Create a new model config for each combination
                        new_model_config = model_config.copy()
                        new_model_config.update({
                            "g_type": g_type,
                            "gnn_model": gnn,
                            "g_params": g_params,
                        })
                        all_model_configs.append(new_model_config)
            if g_type == 'ego':
                pass
       

    # Update eval config with command line arguments
    eval_config = config['eval_config']
    eval_config.update({
        'datasets': ['Crime','Digg'],
        'per_graphs': 2,
        "synthetic_g_params": {
            "nranges": ["30_50","50_100"],
            "m_values": [1,2]
        },
    })
    # 'datasets': ['Crime','Digg'],
    # "nranges": ["30_50","50_100","100_200","200_300","300_400","400_500"],
    # "m_values": [1,2,3,4,5,6]

    print(f"Model Comparison Evaluation")
    print(f"Number of models to compare: {len(all_model_configs)}")
    print(f"Synthetic graphs per config: {eval_config['per_graphs']}")
    print(f"Real datasets: {eval_config['datasets']}")
    
    # Create save directory
    save_result_dir = eval_config['save_result_dir']
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)
    print(f"\nResults will be saved to: {save_result_dir}")
    

    # ---------------------------------------- Evaluate all models ------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaling Models...")
    print(f"{'='*60}")

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
            
            synthetic_results = {}
            real_results = {}
            iters_synthetic_results = {}
            iters_real_results = {} 
            # Evaluate best iter on synthetic graphs
            if args.eval_synthetic:
                synthetic_results = evaluate_model_on_synthetic_graphs(dqn, model_config, eval_config)
            
            # Evaluate best iter on real graphs
            if args.eval_real:
                real_results = evaluate_model_on_real_graphs(dqn, model_config, eval_config, data_config)
            
            # Evaluate all iters, read from exsiting results by testSynthetic and testReal
            if args.eval_all_iters:
                if args.eval_synthetic:
                    iters_synthetic_results = load_synth_csv(model_config,eval_config)
                if args.eval_real:    
                    iters_real_results = load_real_csv(model_config,eval_config)
            
            # Store results
            model_key = f"model_{i+1}"
            all_model_results[model_key] = {
                'config': model_config,
                'synthetic': synthetic_results,
                'real': real_results,
                'iters_synthetic': iters_synthetic_results,
                'iters_real':iters_real_results,
                'best_iter': best_iter
            }
            
        except Exception as e:
            print(f"  ✗ Error evaluating model {i+1}: {e}")
            continue
    

    # Create plots
    print(f"\n{'='*60}")
    print("Creating MaxCC plots...")
    print(f"{'='*60}")

    # ---------------------Plot per-dataset figures (e.g., BA_nrange_30_50_m_1_max_cc_plot.png) , one test graph one png file------------------------------------------
    
    # synth/real_results will be reformated to {synth_dataset:{model_key:model_results}

    # plot synthetic graph comparisons
    if args.eval_synthetic:
        synth_grid = _get_synth_param_grid(eval_config)
        synth_datasets = [g['synth_dataset'] for g in synth_grid]
        synthetic_results = {}
        for synth_dataset in synth_datasets:
            # Collect results for per-dataset across all models
            synthetic_results[synth_dataset] = {}
            for model_key, model_results in all_model_results.items():
                if model_results['synthetic'] and synth_dataset in model_results['synthetic']:
                    synthetic_results[synth_dataset][model_key] = model_results['synthetic'][synth_dataset]
        if synthetic_results:
            plot_max_cc_comparison(synthetic_results, save_result_dir, all_model_configs)
    
    # Plot real graph comparisons
    if args.eval_real:
        real_results = {}
        for real_dataset in eval_config['datasets']:
            # Collect results for this dataset across all models
            real_results[real_dataset] = {}
            for model_key, model_results in all_model_results.items():
                if model_results['real'] and real_dataset in model_results['real']:
                    real_results[real_dataset][model_key] = model_results['real'][real_dataset]
        if real_results:
            plot_max_cc_comparison(real_results, save_result_dir, all_model_configs)
    

    # ---------------------------------Plot per-dataset figures (e.g. synthetic_BA_grouped_max_cc.png) all models on all datasets-------------------------------------------------------
    
    # TODO:maybe need to reformat like plot_max_cc_comparison
    print(f"\nPlotting MaxCC overview for all models...")
    if args.eval_synthetic:
        plot_best_iter_overview(all_model_results, eval_config, save_result_dir, all_model_configs, mode='synthetic')
        if args.eval_all_iters:
            plot_all_iters_overview(all_model_results, eval_config, save_result_dir, all_model_configs, mode='synthetic')
    if args.eval_real:
        plot_best_iter_overview(all_model_results, eval_config, save_result_dir, all_model_configs, mode='real')
        if args.eval_all_iters:
            plot_all_iters_overview(all_model_results, eval_config, save_result_dir, all_model_configs, mode='real')
    
    print(f"\n{'='*60}")
    print("Model comparison evaluation completed!")
    print(f"All plots saved to: {save_result_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

    
    
