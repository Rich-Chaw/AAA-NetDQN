#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from unittest import result
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import json
import argparse

from testUtils import load_config,create_model,detailed_result_dir,load_real_graph,load_real_csv,save_real_results



def load_sol(iter,config):
    """load solution files for a given iteration"""
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    sol_files = []
    
    for dataset in datasets:
        # Check for solution file
        sol_file = f"{save_result_dir}/{dataset}_iter_{iter}.txt"
        if os.path.exists(sol_file):
            sol_files.append(sol_file)
    return sol_files

def get_evaluation_status(existing_results, target_iters, all_datasets):
    """Determine which iterations and datasets need evaluation based on the new results format"""
    needed_iters = []
    needed_datasets_per_iter = {} # {iter_num: [dataset1, dataset2, ...]}
    
    # For each target iteration, check if it's fully evaluated across all datasets
    for iter_num in target_iters:
        datasets_missing_for_this_iter = []
        for dataset in all_datasets:
            # Check if this dataset exists in existing_results and if this iteration is present
            if dataset not in existing_results or iter_num not in existing_results[dataset] or \
               existing_results[dataset][iter_num]['score'] is None or existing_results[dataset][iter_num]['time'] is None:
                datasets_missing_for_this_iter.append(dataset)
        
        if datasets_missing_for_this_iter:
            needed_iters.append(iter_num)
            needed_datasets_per_iter[iter_num] = datasets_missing_for_this_iter
            
    return needed_iters, needed_datasets_per_iter

def merge_results(existing_results, new_results_list, all_datasets):
    """Merge new results into existing results structure"""
    # existing_results: {dataset_name: {iter_num: {'score': score, 'time': time}}}
    # new_results_list: list of {dataset_name: {iter_num: {'score': score, 'time': time}}} for new evaluations

    # Iterate through each new evaluation result (which corresponds to one iter_num and its evaluated datasets)
    for new_eval_for_iter in new_results_list:
        # Each new_eval_for_iter is a dict: {dataset_name: {'score': s, 'time': t}}
        for dataset_name, data in new_eval_for_iter.items():
            iter_num = data.pop('iter') # Extract iter_num, it's added by eval_one_iter_partial
            if dataset_name not in existing_results:
                existing_results[dataset_name] = {}
            existing_results[dataset_name][iter_num] = data

    # Ensure all datasets are present in existing_results structure for completeness
    for dataset in all_datasets:
        if dataset not in existing_results:
            existing_results[dataset] = {}
            
    return existing_results

def print_evaluation_summary(existing_results, target_iters, all_datasets):
    """Print a detailed summary of evaluation status based on the new results format"""
    if not existing_results:
        print(f"✓ No existing results found.")
        print(f"✓ Will evaluate {len(target_iters)} iterations: {target_iters}")
        return
    
    completed_iters = []
    partial_iters = [] # list of (iter_num, missing_datasets)
    missing_iters = []
    
    total_possible_datasets_per_iter = len(all_datasets)

    for iter_num in target_iters:
        missing_datasets_for_this_iter = []
        for dataset in all_datasets:
            if dataset not in existing_results or iter_num not in existing_results[dataset] or \
               existing_results[dataset][iter_num]['score'] is None or existing_results[dataset][iter_num]['time'] is None:
                missing_datasets_for_this_iter.append(dataset)
        
        if not missing_datasets_for_this_iter:
            completed_iters.append(iter_num)
        elif len(missing_datasets_for_this_iter) < total_possible_datasets_per_iter:
            partial_iters.append((iter_num, missing_datasets_for_this_iter))
        else:
            # All datasets are missing for this iteration
            missing_iters.append(iter_num)
    
    print(f"✓ Evaluation Summary:")
    print(f"  - Completed iterations: {len(completed_iters)}")
    if completed_iters:
        print(f"    {completed_iters[:5]}{'...' if len(completed_iters) > 5 else ''}")
    
    print(f"  - Partial iterations: {len(partial_iters)}")
    for iter_num, missing in partial_iters[:3]:  # Show first 3
        print(f"    Iteration {iter_num}: missing {len(missing)}/{total_possible_datasets_per_iter} datasets ({missing})")
    if len(partial_iters) > 3:
        print(f"    ... and {len(partial_iters) - 3} more partial iterations")
    
    # Show detailed missing datasets for each iteration
    if partial_iters:
        print(f"  - Detailed missing datasets for partial iterations:")
        for iter_num, missing in partial_iters:
            print(f"    Iteration {iter_num}: {missing}")
    
    print(f"  - Missing iterations (no datasets evaluated): {len(missing_iters)}")
    if missing_iters:
        print(f"    {missing_iters[:5]}{'...' if len(missing_iters) > 5 else ''}")
    
    total_to_evaluate = len(missing_iters) + len(partial_iters)
    print(f"  - Total iterations with pending evaluation: {total_to_evaluate}")

def print_evaluation_plan(needed_iters, needed_datasets_per_iter, datasets):
    """Print detailed plan of what will be evaluated"""
    print(f"\n" + "="*60)
    print("EVALUATION PLAN")
    print("="*60)
    
    if not needed_iters:
        print("✓ No evaluation needed - all iterations are complete!")
        return
    
    print(f"✓ Will evaluate {len(needed_iters)} iterations:")
    
    for iter_num in needed_iters:
        needed_datasets = needed_datasets_per_iter.get(iter_num, datasets)
        if len(needed_datasets) == len(datasets):
            print(f"  - Iteration {iter_num}: ALL datasets ({len(needed_datasets)} datasets)")
        else:
            print(f"  - Iteration {iter_num}: {needed_datasets} ({len(needed_datasets)} datasets)")
    
    total_datasets_to_evaluate = sum(len(needed_datasets_per_iter.get(iter, datasets)) for iter in needed_iters)
    print(f"\n✓ Total evaluations: {total_datasets_to_evaluate} dataset-iteration combinations")
    print("="*60)


def eval_one_iter_partial(iter_num, config, datasets_to_evaluate, save_sol=False):
    """Evaluate a single iteration checkpoint on specific datasets only"""
    print(f"\nEvaluating iteration {iter_num}...")
    
    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']

    # Create model for this iteration
    dqn = create_model(model_config, iter_num)
    
    step_ratio = eval_config['step_ratio']
    strategy_id = eval_config['strategy_id']
    
    # Store results for this iteration for datasets_to_evaluate
    newly_evaluated_results = {}

    if not datasets_to_evaluate:
        print(f"  ✓ No datasets specified for evaluation in iteration {iter_num}.")
        return newly_evaluated_results
        
    print(f"  ✓ Evaluating {len(datasets_to_evaluate)} datasets: {datasets_to_evaluate}")
    
    # Evaluate each specified dataset
    for dataset in datasets_to_evaluate:
        print(f"    Evaluating dataset: {dataset}")
        dataset_dir = os.path.join(data_config['dataset_dir'],"real")
        g_test = load_real_graph(dataset,dataset_dir)
        if g_test is None:
            print(f"      Warning: Could not load graph for {dataset}")
            newly_evaluated_results[dataset] = {'score': None, 'time': None}
            continue
        
        # Get solution, and save in temp_result_file
        if save_sol == False:
            temp_sol_file = f"temp_{dataset}_{iter_num}.txt"
        else:
            save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
            temp_sol_file = f"{save_result_dir}/{dataset}_iter_{iter_num}.txt"

        try:
            sol, sol_time = dqn.EvaluateRealData(g_test, temp_sol_file, step_ratio)
            # Evaluate solution
            t1 = time.time()
            score, MaxCCList = dqn.EvaluateSol(g_test, temp_sol_file, strategy_id, reInsertStep=0.001)
            eval_time = time.time() - t1
            total_time = sol_time + eval_time
            
            newly_evaluated_results[dataset] = {'score': score, 'time': total_time}
            print(f"      Score: {score:.6f}, Total time: {total_time:.2f}s")
            
            # Clean up temp file
            if os.path.exists(temp_sol_file) and save_sol == False:
                os.remove(temp_sol_file)
                
        except Exception as e:
            print(f"      Error evaluating {dataset}: {e}")
            newly_evaluated_results[dataset] = {'score': None, 'time': None}
    
    return newly_evaluated_results

def eval_all_iters(config):
    """Evaluate all checkpoints from min_iter to max_iter, skipping already evaluated iterations"""
    
    eval_config = config['eval_config']
    model_config = config['model_config']

    min_iter = eval_config['min_iter']
    max_iter = eval_config['max_iter']
    iter_step = eval_config['iter_step']
    datasets = eval_config['datasets']
    
    # Generate target iteration list
    target_iters = list(range(min_iter, max_iter + 1, iter_step))
    print(f"Target iterations: {len(target_iters)} iterations from {target_iters[0]} to {target_iters[-1]} (step: {iter_step})")
    
    # Load existing results (new format)
    existing_results = load_real_csv(model_config,eval_config)
    
    # Determine what needs to be evaluated
    needed_iters, needed_datasets_per_iter = get_evaluation_status(existing_results, target_iters, datasets)

    # Print evaluation summary
    print_evaluation_summary(existing_results, target_iters, datasets)
    
    if not needed_iters:
        print("✓ All iterations already evaluated! No new evaluation needed.")
        print("skipping saving results...")
        return existing_results
    
    print(f"✓ Need to evaluate {len(needed_iters)} iterations: {needed_iters}")
    
    # Print detailed evaluation plan
    print_evaluation_plan(needed_iters, needed_datasets_per_iter, datasets)
    
    # Results storage for newly evaluated data
    new_results_list = [] # List of dicts, each dict is {dataset_name: {'score':s, 'time':t}, 'iter': iter_num}
    
    # Evaluate each needed iteration
    for iter_num in tqdm(needed_iters, desc="Evaluating iterations"):
        try:
            # Get the specific datasets that need evaluation for this iteration
            datasets_to_evaluate = needed_datasets_per_iter.get(iter_num, datasets)
            
            # Call eval_one_iter_partial to get results for this iteration and its needed datasets
            # This function returns {dataset_name: {'score': s, 'time': t}}
            iter_new_results = eval_one_iter_partial(iter_num, config, datasets_to_evaluate, save_sol=False)
            
            # Add iter_num to each dataset's result for easier merging
            for dataset_name in iter_new_results:
                iter_new_results[dataset_name]['iter'] = iter_num
                
            new_results_list.append(iter_new_results)
                  
        except Exception as e:
            print(f"✗ Error evaluating iteration {iter_num}: {e}")
            # For a failed iteration, ensure we still record it as missing for affected datasets
            failed_iter_results = {}
            for dataset in datasets_to_evaluate:
                failed_iter_results[dataset] = {'score': None, 'time': None, 'iter': iter_num}
            new_results_list.append(failed_iter_results)
    
    # Merge new results with existing results
    all_results = merge_results(existing_results, new_results_list, datasets)
    # save results
    save_real_results(all_results, model_config, eval_config)
    
    return all_results


def main():
    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
 
    # parse args
    parser = argparse.ArgumentParser(description='Evaluate ONE model on real datasets')
    parser.add_argument("--model_path", type=str, help="Path to model directory (e.g., ./models/BA_nrange_30_50_m_3)")
    parser.add_argument("--min_iter", type=int, default=0, help="Minimum iteration to evaluate")
    parser.add_argument("--max_iter", type=int, default=6000, help="Maximum iteration to evaluate")
    parser.add_argument("--iter_step", type=int, default=300, help="Iteration step size")
    parser.add_argument("--eval_all_iters", action="store_true")
    parser.add_argument("--eval_iter", type=int)
    parser.add_argument("--save_sol_only", action="store_true", help="save sol to <dataset>_iter.txt,not update csv")
    args = parser.parse_args()

    # Update config with command line arguments
    eval_config = config['eval_config']
    eval_config.update({
        'min_iter': args.min_iter,
        'max_iter': args.max_iter,
        'iter_step': args.iter_step,
        "datasets": ["Digg"]
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
    
    print(f"\n{'-'*10}Real Dataset Evaluation{'.'*10}")
    print(f"Target model: {model_config['g_type']}_nrange_{model_config['g_params']['nrange']}_m_{model_config['g_params']['m']}")

    # eval_all_iters
    if args.eval_all_iters:
        print("\neval_all_iters is True, running comprehensive evaluation...")
        
        try:
            from testUtils import plot_val_eval_scores
            all_results = eval_all_iters(config)
            # plot_val_eval_scores(config) # This function will need to be updated to handle new result format
            return
        except ImportError:
            print("Warning: plot_val_eval_scores not found or import error, skipping plot")
            
    
    if args.eval_iter:
        # eval one iter
        if args.eval_iter < 0:
            # find best model iter using Dqn.findModel
            dqn = create_model(config['model_config'])
            print("eval_iter = None, find best iter by dqn.findModel")
            best_ckpt_file = dqn.findModel()
            dqn.LoadModel(best_ckpt_file)
            best_iter = int(best_ckpt_file.split('.ckpt')[0].split('_')[-1])
            args.eval_iter = best_iter
        print("\neval_all_iter is False, evaluate iteration %d"%args.eval_iter)

        target_iters = [args.eval_iter]
        all_datasets = eval_config['datasets']
        
        if args.save_sol_only:
            # When save_sol, do not save results to prevent overwrite of CSVs, just solutions
            for iter_num in target_iters:
                # Determine which datasets still need solution files
                existing_sols = load_sol(iter_num, config)
                evaluated_datasets_for_sols = [os.path.basename(s).split(f'_iter_{iter_num}.txt')[0].split('_')[0] for s in existing_sols]
                needed_datasets_for_sols = [dataset for dataset in all_datasets if dataset not in evaluated_datasets_for_sols]
                
                if needed_datasets_for_sols:
                    print(f"Evaluating iteration {iter_num} to save solutions for datasets: {needed_datasets_for_sols}")
                    eval_one_iter_partial(iter_num, config, needed_datasets_for_sols, save_sol=True)
                else:
                    print(f"All solution files for iteration {iter_num} already exist, skipping.")
            
        else:
            # Normal evaluation for a specified iteration, including saving results
            existing_results = load_real_csv(model_config,eval_config)
            needed_iters, needed_datasets_per_iter = get_evaluation_status(existing_results, target_iters, all_datasets)
            
            if not needed_iters:
                print("No new evaluation needed for specified iteration, skipping evaluation and save_results...")
            else:
                new_results_list = []
                for iter_num in needed_iters:
                    datasets_to_evaluate = needed_datasets_per_iter.get(iter_num, all_datasets)
                    
                    # Perform evaluation
                    iter_new_results = eval_one_iter_partial(iter_num, config, datasets_to_evaluate, save_sol=True)
                    
                    # Add iter_num to each dataset's result
                    for dataset_name in iter_new_results:
                        iter_new_results[dataset_name]['iter'] = iter_num
                        
                    new_results_list.append(iter_new_results)

                # Merge and save results
                all_results = merge_results(existing_results, new_results_list, all_datasets)
                save_real_results(all_results, model_config, eval_config)


if __name__=="__main__":
    main()
