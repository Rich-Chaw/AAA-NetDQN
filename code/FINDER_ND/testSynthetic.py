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
from testUtils import load_config, create_model, _get_synth_param_grid,load_synthetic_graphs,detailed_result_dir

from testUtils import visualize_graph_dismantling,create_dismantling_animation,plot_max_cc_curve,plot_max_cc_lists


def load_existing_synthetic_results(eval_config, model_config):
    """
    Load existing evaluation results from CSV files to avoid re-evaluation
    
    Returns:
    - dict: existing results organized by iteration -> config_key -> results
    """
    save_result_dir = detailed_result_dir(model_config, eval_config, mode='synthetic')
    
    if not os.path.exists(save_result_dir):
        print(f"  No existing results directory found: {save_result_dir}")
        return {}
    
    existing_results = {}
    
    # Check each configuration for existing CSV files using parameter grid
    param_grid = _get_synth_param_grid(eval_config)
    for entry in param_grid:
        config_key = entry['config_key']
        csv_filename = entry['filename']
        csv_path = os.path.join(save_result_dir, csv_filename)
            
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"  ✓ Found existing results: {csv_filename} ({len(df)} rows)")
                
                # Convert DataFrame to results format
                for _, row in df.iterrows():
                    iter_num = int(row['iter'])
                    if iter_num not in existing_results:
                        existing_results[iter_num] = {}
                    
                    existing_results[iter_num][config_key] = {
                        'score': float(row['score']) if pd.notna(row['score']) else None,
                        'time': float(row['time']) if pd.notna(row['time']) else None
                    }
                    
            except Exception as e:
                print(f"  ⚠ Warning: Could not read {csv_filename}: {e}")
        else:
            print(f"  - No existing results for {config_key}")
    
    if existing_results:
        print(f"  ✓ Loaded {len(existing_results)} existing iterations")
        existing_iters = sorted(existing_results.keys())
        print(f"    Iterations found: {existing_iters[:5]}{'...' if len(existing_iters) > 5 else ''}")
    
    return existing_results

def get_evaluation_status_synthetic(existing_results, target_iters, eval_config):
    """
    Determine which iterations and configurations need evaluation
    
    Returns:
    - list: iterations that need evaluation
    - dict: for each iteration, which configs need evaluation
    """
    needed_iters = []
    needed_configs_per_iter = {}
    
    # Build the full configuration key set from grid
    synth_grid = _get_synth_param_grid(eval_config)
    config_keys = [g['config_key'] for g in synth_grid]

    for iter_num in target_iters:
        if iter_num not in existing_results:
            # New iteration - need all configs
            needed_iters.append(iter_num)
            needed_configs_per_iter[iter_num] = list(config_keys)
            continue
        
        # Check which configs are missing for this iteration
        existing_configs = existing_results[iter_num]
        missing_configs = []
        
        for config_key in config_keys:
            if (config_key not in existing_configs or 
                existing_configs[config_key]['score'] is None or 
                existing_configs[config_key]['time'] is None):
                missing_configs.append(config_key)
        
        if missing_configs:
            needed_iters.append(iter_num)
            needed_configs_per_iter[iter_num] = missing_configs
    
    return needed_iters, needed_configs_per_iter

def print_evaluation_summary_synthetic(existing_results, target_iters, eval_config):
    """
    Print a detailed summary of synthetic evaluation status
    """
    if not existing_results:
        print(f"✓ No existing results found")
        print(f"✓ Will evaluate {len(target_iters)} iterations: {target_iters[:5]}{'...' if len(target_iters) > 5 else ''}")
        return
    
    # existing_results is already in the correct format: {iter_num: config_results}
    
    completed_iters = []
    partial_iters = []
    missing_iters = []
    
    synth_grid = _get_synth_param_grid(eval_config)
    total_configs = len(synth_grid)
    config_keys = [g['config_key'] for g in synth_grid]
    
    for iter_num in target_iters:
        if iter_num not in existing_results:
            missing_iters.append(iter_num)
            continue
        
        result = existing_results[iter_num]
        all_complete = True
        missing_configs = []
        
        for config_key in config_keys:
            if (config_key not in result or 
                result[config_key]['score'] is None or 
                result[config_key]['time'] is None):
                all_complete = False
                missing_configs.append(config_key)
        
        if all_complete:
            completed_iters.append(iter_num)
        else:
            partial_iters.append((iter_num, missing_configs))
    
    print(f"✓ Evaluation Summary:")
    print(f"  - Completed iterations: {len(completed_iters)}")
    if completed_iters:
        print(f"    {completed_iters[:5]}{'...' if len(completed_iters) > 5 else ''}")
    
    print(f"  - Partial iterations: {len(partial_iters)}")
    for iter_num, missing in partial_iters[:3]:  # Show first 3
        print(f"    Iteration {iter_num}: missing {len(missing)}/{total_configs} configs")
    if len(partial_iters) > 3:
        print(f"    ... and {len(partial_iters) - 3} more")
    
    print(f"  - Missing iterations: {len(missing_iters)}")
    if missing_iters:
        print(f"    {missing_iters[:5]}{'...' if len(missing_iters) > 5 else ''}")
    
    total_to_evaluate = len(missing_iters) + len(partial_iters)
    print(f"  - Total iterations to evaluate: {total_to_evaluate}")
    
    if total_to_evaluate > 0:
        total_configs_to_evaluate = 0
        for iter_num in missing_iters:
            total_configs_to_evaluate += total_configs
        for iter_num, missing in partial_iters:
            total_configs_to_evaluate += len(missing)
        print(f"  - Total configurations to evaluate: {total_configs_to_evaluate}")

def print_evaluation_progress(completed, total, current_iter, current_config):
    """
    Print evaluation progress information
    """
    progress = (completed / total) * 100 if total > 0 else 0
    print(f"  Progress: {completed}/{total} ({progress:.1f}%) - Current: iter {current_iter}, {current_config}")

def evaluate_checkpoint_on_synthetic_datasets(dqn, checkpoint_iter, eval_config, existing_results=None):
    """
    Evaluate a single checkpoint on all synthetic dataset configurations
    
    Parameters:
    - dqn: GraphDQN model instance
    - checkpoint_iter: iteration number to evaluate
    - eval_config: evaluation configuration
    - existing_results: existing results to avoid re-evaluation

    Returns:
    - dict: results for each dataset configuration
    """
    print(f"  Evaluating checkpoint {checkpoint_iter}...")

    # Load the specific checkpoint
    dqn.LoadModel(f"{dqn.embeddingMethod}_iter_{checkpoint_iter}.ckpt")
    
    results = {}
    
    # Check if we have existing results for this iteration
    existing_iter_results = existing_results.get(checkpoint_iter, {}) if existing_results else {}
    
    # Evaluate on each synthetic dataset configuration
    synth_grid = _get_synth_param_grid(eval_config)
    for entry in synth_grid:
        config_key = entry['config_key']
        params = entry['params']
        
        # Check if we already have results for this config
        if config_key in existing_iter_results:
            existing_result = existing_iter_results[config_key]
            if (existing_result['score'] is not None and 
                existing_result['time'] is not None):
                print(f"    ✓ Skipping {config_key} (already evaluated)")
                results[config_key] = existing_result
                continue
        
        print(f"    Testing {config_key}...")
        
        try:
            # Load synthetic graphs for this configuration

            graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                    g_num = eval_config["per_graphs"],
                                    **params)

            if not graphs:
                print(f"      Warning: No graphs found for {config_key}")
                results[config_key] = {'score': None, 'time': None}
                continue
            
            # Evaluate on all graphs and get average
            score_mean, score_std, time_mean, time_std = dqn.Evaluate(graphs)
            score = score_mean
            total_time = time_mean
        
                    
            results[config_key] = {
                'score': score,
                'time': total_time
            }
                
        except Exception as e:
            print(f"      ✗ Error evaluating {config_key}: {e}")
            results[config_key] = {'score': None, 'time': None}
    
    return results

def save_synthetic_results(all_results, eval_config, model_config):
    """
    Save synthetic evaluation results to CSV files, merging with existing results
    
    Parameters:
    - all_results: dict with iteration -> results mapping
    - eval_config: evaluation configuration
    - model_config: model configuration
    """
    # Create save directory structure
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='synthetic')
    
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)
    
    print(f"\nSaving results to: {save_result_dir}")
    
    # Save results for each dataset configuration
    synth_grid = _get_synth_param_grid(eval_config)
    for entry in synth_grid:
        config_key = entry['config_key']
        csv_filename = entry['filename']
        csv_path = os.path.join(save_result_dir, csv_filename)
            
        # Load existing data if file exists
        existing_data = {'iter': [], 'score': [], 'time': []}
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                existing_data = {
                    'iter': existing_df['iter'].tolist(),
                    'score': existing_df['score'].tolist(),
                    'time': existing_df['time'].tolist()
                }
                print(f"  ✓ Loaded existing data: {csv_filename} ({len(existing_df)} rows)")
            except Exception as e:
                print(f"  ⚠ Warning: Could not read existing {csv_filename}: {e}")
        
        # Prepare new data
        new_data = {
            'iter': [],
            'score': [],
            'time': []
        }
        
        for iter_num in sorted(all_results.keys()):
            if all_results[iter_num] and config_key in all_results[iter_num]:
                result = all_results[iter_num][config_key]
                # Only add if not already in existing data
                if iter_num not in existing_data['iter']:
                    new_data['iter'].append(iter_num)
                    new_data['score'].append(result['score'])
                    new_data['time'].append(result['time'])
        
        # Merge existing and new data
        merged_data = {
            'iter': existing_data['iter'] + new_data['iter'],
            'score': existing_data['score'] + new_data['score'],
            'time': existing_data['time'] + new_data['time']
        }
        
        # Create DataFrame and save
        if merged_data['iter']:
            df = pd.DataFrame(merged_data)
            # Sort by iteration number
            df = df.sort_values('iter').reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            
            if new_data['iter']:
                print(f"  ✓ Updated {csv_filename}: {len(existing_data['iter'])} existing + {len(new_data['iter'])} new = {len(merged_data['iter'])} total")
            else:
                print(f"  ✓ No new data for {csv_filename} (kept {len(existing_data['iter'])} existing)")
        else:
            print(f"  ✗ No data for {csv_filename}")


def main():
    # Load configuration
    config = load_config()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate models on synthetic datasets')
    parser.add_argument("--model_path", type=str, help="Path to model directory (e.g., ./models/BA_nrange_30_50_m_3)")
    parser.add_argument("--min_iter", type=int, default=0, help="Minimum iteration to evaluate")
    parser.add_argument("--max_iter", type=int, default=6000, help="Maximum iteration to evaluate")
    parser.add_argument("--iter_step", type=int, default=300, help="Iteration step size")
    parser.add_argument("--per_graphs", type=int, default=100, help="Number of graphs per configuration")
    parser.add_argument("--eval_iter",type=int)
    parser.add_argument("--eval_all_iters", action="store_true")

    args = parser.parse_args()
    
    # Update config with command line arguments
    eval_config = config['eval_config']
    eval_config.update({
        'min_iter': args.min_iter,
        'max_iter': args.max_iter,
        'iter_step': args.iter_step,
        'per_graphs': args.per_graphs
    })
    
    

    # Update model config if model_path is provided
    model_config = config['model_config']
    if args.model_path:
        # Parse model path to extract parameters
        path_parts = args.model_path.split('/')[-1].split('_')
        if len(path_parts) >= 6:
            model_config['g_type'] = path_parts[0]
            model_config['g_params']['nrange'] = path_parts[2]+'_'+path_parts[3]
            model_config['g_params']['m'] = int(path_parts[5])
            print(f"✓ Updated model config: {model_config}")
    
    print(f"\nSynthetic Dataset Evaluation")
    print(f"Target model: {model_config['g_type']}_nrange_{model_config['g_params']['nrange']}_m_{model_config['g_params']['m']}")
    

    print(f"\nStarting evaluation...")
    if args.eval_all_iters:
        # --------------------------eval all iters ----------------------------------
        # Generate target iterations
        target_iters = list(range(eval_config['min_iter'], eval_config['max_iter'] + 1, eval_config['iter_step']))
        # Create model instance
        dqn = create_model(model_config)
        
        # Load existing results
        existing_results = load_existing_synthetic_results(eval_config, model_config)
        
        # Print evaluation summary
        print_evaluation_summary_synthetic(existing_results, target_iters, eval_config)
        
        # Determine which iterations and configs need evaluation
        needed_iters, needed_configs_per_iter = get_evaluation_status_synthetic(existing_results, target_iters, eval_config)
        
        if not needed_iters:
            print("✓ All iterations already evaluated! No new evaluation needed.")
            all_results = existing_results
        else:
            print(f"\nIterations to evaluate: {needed_iters}")
            total_configs_to_evaluate = sum(len(configs) for configs in needed_configs_per_iter.values())
            print(f"Total configurations to evaluate: {total_configs_to_evaluate}")
            
            # Evaluate all checkpoints
            print(f"\nStarting evaluation...")
            print(f"Target iterations: {eval_config['min_iter']} to {eval_config['max_iter']} (step: {eval_config['iter_step']})")
            grid = _get_synth_param_grid(eval_config)
            print(f"Dataset configurations: {len(grid)} total")
            print(f"Graphs per: {eval_config['per_graphs']}")
            
            # Start with existing results
            all_results = existing_results.copy()
            
            # Track progress
            total_evaluations = len(needed_iters)
            completed_evaluations = 0
            
            for iter_num in tqdm(needed_iters, desc="Evaluating checkpoints"):
                try:
                    print(f"\nEvaluating iteration {iter_num}...")
                    results = evaluate_checkpoint_on_synthetic_datasets(dqn, iter_num, eval_config, existing_results)
                    if results:
                        all_results[iter_num] = results
                    else:
                        all_results[iter_num] = None
                    
                    completed_evaluations += 1
                    print_evaluation_progress(completed_evaluations, total_evaluations, iter_num, "completed")
                    
                except Exception as e:
                    print(f"✗ Error evaluating iteration {iter_num}: {e}")
                    all_results[iter_num] = None
                    completed_evaluations += 1
        
        # Save results (this will merge existing and new results)
        save_synthetic_results(all_results, eval_config, model_config)
        return
    
    if args.eval_iter:
        # --------------------------eval specified iter, draw sol and CC curve ----------------------------------
        if args.eval_iter < 0:
            dqn = create_model(model_config,iter)
            print("eval_iter = None, find best iter by dqn.findModel")
            best_ckpt_file = dqn.findModel()
            best_iter = int(best_ckpt_file.split('.ckpt')[0].split('_')[-1])
            args.eval_iter = best_iter
        
        dqn = create_model(model_config,args.eval_iter)

        synth_grid = _get_synth_param_grid(eval_config)
        for entry in synth_grid:
            params = entry['params']
            config_key = entry['config_key']
    
            graphs = load_synthetic_graphs(g_type = eval_config["synthetic_g_type"],
                                            g_num = eval_config["per_graphs"],
                                            **params)

            all_MaxCCList = []
            save_result_dir = detailed_result_dir(model_config,eval_config,mode='synthetic')
            for i,g in enumerate(graphs):
                general_result_file = os.path.join(save_result_dir,f"{eval_config['synthetic_g_type']}_{config_key}_g_{i}_iter_{iter}")
                
                # Get sol and MaxCCList
                temp_sol_file = f"{general_result_file}.txt"
                # sol = effective solution = nodes to remove
                sol, sol_time = dqn.EvaluateRealData(g, temp_sol_file, eval_config['step_ratio'])
                # print(sol)
                # Evaluate sol , solution = sol_reinsert + sol_left
                score, MaxCCList = dqn.EvaluateSol(g, temp_sol_file, eval_config['strategy_id'], reInsertStep=0.001)
                all_MaxCCList.append(MaxCCList)

                # Visualize dismantling process, the funcion has tested, however no need to do here
                if sol and len(sol) > 0:
                    print(f"\nAnalyzing dismantling solution for graph {i}...")
                    try:
                        viz_file_path = f"{general_result_file}_dismantling.png"
                        visualize_graph_dismantling(g, sol, viz_file_path)

                        anim_file_path = f"{general_result_file}_dismantling_anim.gif"
                        create_dismantling_animation(g, sol, anim_file_path)

                        plot_file_path = f"{general_result_file}_MaxCC_curve.png"
                        plot_max_cc_curve(MaxCCList, plot_file_path)

                        print(f"  ✓ All visualizations completed for graph {i}")
                    except Exception as e:
                        print(f"  ✗ Error creating visualizations for graph {i}: {e}")
                else:
                    print(f"  ⚠ No solution found for graph {i}, skipping visualizations")
                
            # plot MaxCC curve across graphs
            compre_file_path = os.path.join(save_result_dir,f"{eval_config['synthetic_g_type']}_{config_key}_iter_{iter}_MaxCC_comparison.png")
            plot_max_cc_lists(all_MaxCCList,compre_file_path)
 




if __name__ == "__main__":
    main()
