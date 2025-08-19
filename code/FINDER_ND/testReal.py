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

from testUtils import load_config,create_model,detailed_result_dir,load_real_graph

def load_csv(config):
    """Load existing evaluation results from individual dataset CSV files
        for datasets in eval_config
    """
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    
    print(f"Looking for existing results in: {save_result_dir}")
    
    existing_results = []
    
    # Check if individual dataset files exist
    available_datasets = []
    for dataset in datasets:
        dataset_file = f"{save_result_dir}/{dataset}.csv"
        if os.path.exists(dataset_file):
            available_datasets.append(dataset)
            print(f"  Found dataset file: {dataset_file}")
        else:
            print(f"  Dataset file not found: {dataset_file}")
    
    if not available_datasets:
        print("✗ No existing results found, starting fresh evaluation")
        return existing_results
    
    print(f"✓ Found {len(available_datasets)} dataset files")
    
    # Load data from each available dataset file
    dataset_data = {}
    for dataset in available_datasets:
        try:
            dataset_file = f"{save_result_dir}/{dataset}.csv"
            df = pd.read_csv(dataset_file)
            
            # Verify required columns exist
            if not all(col in df.columns for col in ['iter', 'score', 'time']):
                print(f"  Warning: {dataset}.csv missing required columns. Found: {list(df.columns)}")
                continue
            
            # Clean data - remove rows with NaN values
            df_clean = df.dropna(subset=['iter', 'score', 'time'])
            if df_clean.empty:
                print(f"  Warning: {dataset}.csv has no valid data after cleaning")
                continue
            
            dataset_data[dataset] = df_clean
            print(f"  Loaded {dataset}.csv with {len(df_clean)} valid rows")
            
        except Exception as e:
            print(f"  Error loading {dataset}.csv: {e}")
            continue
    
    if not dataset_data:
        print("✗ No valid dataset files could be loaded")
        return existing_results
    
    # Combine data from all datasets
    all_iters = set()
    for dataset, df in dataset_data.items():
        all_iters.update(df['iter'].tolist())
    
    all_iters = sorted(list(all_iters))
    print(f"✓ Found {len(all_iters)} unique iterations across all datasets")
    
    # Create results structure
    for iter_num in all_iters:
        iter_result = {
            'iter': int(iter_num),
            'scores': {},
            'times': {}
        }
        
        for dataset in datasets:
            if dataset in dataset_data:
                df = dataset_data[dataset]
                iter_row = df[df['iter'] == iter_num]
                if not iter_row.empty:
                    iter_result['scores'][dataset] = float(iter_row.iloc[0]['score'])
                    iter_result['times'][dataset] = float(iter_row.iloc[0]['time'])
                else:
                    iter_result['scores'][dataset] = None
                    iter_result['times'][dataset] = None
            else:
                iter_result['scores'][dataset] = None
                iter_result['times'][dataset] = None
        
        existing_results.append(iter_result)
    
    print(f"✓ Successfully loaded {len(existing_results)} evaluation results")
    if existing_results:
        print(f"  Iterations found: {[r['iter'] for r in existing_results]}")
    
    return existing_results


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

def get_evaluation_status(existing_results, target_iters, datasets):
    """Determine which iterations and datasets need evaluation"""
    # Find iterations that need evaluation and their missing datasets
    needed_iters = []
    needed_datasets_per_iter = {}
    
    if not existing_results:
        for iter_num in target_iters:
            needed_datasets_per_iter[iter_num] = datasets.copy()
        return target_iters, needed_datasets_per_iter
    
    # Create lookup for existing results
    existing_lookup = {}
    for result in existing_results:
        iter_num = result['iter']
        existing_lookup[iter_num] = result
    
    for iter_num in target_iters:
        if iter_num not in existing_lookup:
            # New iteration - need all datasets
            needed_iters.append(iter_num)
            needed_datasets_per_iter[iter_num] = datasets.copy()
            continue
        
        # Check which datasets are missing for this iteration
        result = existing_lookup[iter_num]
        missing_datasets = []
        for dataset in datasets:
            if result['scores'].get(dataset) is None or result['times'].get(dataset) is None:
                missing_datasets.append(dataset)
        
        if missing_datasets:
            # Incomplete iteration - need missing datasets
            needed_iters.append(iter_num)
            needed_datasets_per_iter[iter_num] = missing_datasets
    
    return needed_iters, needed_datasets_per_iter

def merge_results(existing_results, new_results):
    """Merge new results with existing results"""
    if not existing_results:
        return new_results
    
    # Create lookup for existing results
    existing_lookup = {result['iter']: result for result in existing_results}
    
    # Merge new results
    for new_result in new_results:
        iter_num = new_result['iter']
        if iter_num in existing_lookup:
            # Update existing result with new data
            existing = existing_lookup[iter_num]
            for dataset in new_result['scores']:
                if new_result['scores'][dataset] is not None:
                    existing['scores'][dataset] = new_result['scores'][dataset]
                if new_result['times'][dataset] is not None:
                    existing['times'][dataset] = new_result['times'][dataset]
        else:
            # Add new result
            existing_results.append(new_result)
    
    # Sort by iteration number
    existing_results.sort(key=lambda x: x['iter'])
    
    return existing_results

def print_evaluation_summary(existing_results, target_iters, datasets):
    """Print a detailed summary of evaluation status"""
    if not existing_results:
        print(f"✓ No existing results found")
        print(f"✓ Will evaluate {len(target_iters)} iterations: {target_iters}")
        return
    
    # Create lookup for existing results
    existing_lookup = {result['iter']: result for result in existing_results}
    
    completed_iters = []
    partial_iters = []
    missing_iters = []
    
    for iter_num in target_iters:
        if iter_num not in existing_lookup:
            missing_iters.append(iter_num)
            continue
        
        result = existing_lookup[iter_num]
        all_complete = True
        missing_datasets = []
        
        for dataset in datasets:
            if result['scores'].get(dataset) is None or result['times'].get(dataset) is None:
                all_complete = False
                missing_datasets.append(dataset)
        
        if all_complete:
            completed_iters.append(iter_num)
        else:
            partial_iters.append((iter_num, missing_datasets))
    
    print(f"✓ Evaluation Summary:")
    print(f"  - Completed iterations: {len(completed_iters)}")
    if completed_iters:
        print(f"    {completed_iters[:5]}{'...' if len(completed_iters) > 5 else ''}")
    
    print(f"  - Partial iterations: {len(partial_iters)}")
    for iter_num, missing in partial_iters[:3]:  # Show first 3
        print(f"    Iteration {iter_num}: missing {missing}")
    if len(partial_iters) > 3:
        print(f"    ... and {len(partial_iters) - 3} more")
    
    # Show detailed missing datasets for each iteration
    if partial_iters:
        print(f"  - Detailed missing datasets:")
        for iter_num, missing in partial_iters:
            print(f"    Iteration {iter_num}: {missing}")
    
    print(f"  - Missing iterations: {len(missing_iters)}")
    if missing_iters:
        print(f"    {missing_iters[:5]}{'...' if len(missing_iters) > 5 else ''}")
    
    total_to_evaluate = len(missing_iters) + len(partial_iters)
    print(f"  - Total iterations to evaluate: {total_to_evaluate}")

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


def eval_one_iter_partial(iter, config, iter_results=None, specified_datasets=None, save_sol=False):
    """Evaluate a single iteration checkpoint on specific datasets only"""
    print(f"\nEvaluating iteration {iter}...")
    
    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']

    # Create model for this iteration
    dqn = create_model(model_config, iter)
    
    all_datasets = eval_config['datasets']
    step_ratio = eval_config['step_ratio']
    strategy_id = eval_config['strategy_id']
    # dataset_dir = data_config['dataset_dir']
    
    # Initialize results
    if iter_results:
        iter_results = iter_results.copy()
        print(f"  Using existing results for iteration {iter}")
    else:
        iter_results = {
            'iter': iter,
            'scores': {},
            'times': {}
        }
    
    # Determine which datasets to evaluate
    if specified_datasets is None:
        # If no specific datasets provided, evaluate all missing ones
        datasets_eval = []
        datasets_skipped = []
        for dataset in all_datasets:
            if (iter_results['scores'].get(dataset) is None or 
                iter_results['times'].get(dataset) is None):
                datasets_eval.append(dataset)
            else:
                datasets_skipped.append(dataset)
    else:
        # Use the specific datasets that need evaluation
        datasets_eval = specified_datasets
        datasets_skipped = [d for d in all_datasets if d not in specified_datasets]
    
    if not datasets_eval:
        print(f"  ✓ All datasets already evaluated for iteration {iter}")
        return iter_results
    
    print(f"  ✓ Evaluating {len(datasets_eval)} datasets: {datasets_eval}")
    if datasets_skipped:
        print(f"    Skipped: {datasets_skipped}")
    
    # Evaluate each specified dataset
    for dataset in datasets_eval:
        print(f"    Evaluating dataset: {dataset}")
        dataset_dir = os.path.join(data_config['dataset_dir'],"real")
        g_test = load_real_graph(dataset,dataset_dir)
        if g_test is None:
            print(f"      Warning: Could not load graph for {dataset}")
            iter_results['scores'][dataset] = None
            iter_results['times'][dataset] = None
            continue
        
        # Get solution, and save in temp_result_file
        if save_sol == False:
            temp_sol_file = f"temp_{dataset}_{iter}.txt"
        else:
            save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
            temp_sol_file = f"{save_result_dir}/{dataset}_iter_{iter}.txt"

        try:
            sol, sol_time = dqn.EvaluateRealData(g_test, temp_sol_file, step_ratio)
            # Evaluate solution
            t1 = time.time()
            score, MaxCCList = dqn.EvaluateSol(g_test, temp_sol_file, strategy_id, reInsertStep=0.001)
            eval_time = time.time() - t1
            total_time = sol_time + eval_time
            
            iter_results['scores'][dataset] = score
            iter_results['times'][dataset] = total_time
            print(f"      Score: {score:.6f}, Total time: {total_time:.2f}s")
            
            # Clean up temp file
            if os.path.exists(temp_sol_file) and save_sol == False:
                os.remove(temp_sol_file)
                
        except Exception as e:
            print(f"      Error evaluating {dataset}: {e}")
            iter_results['scores'][dataset] = None
            iter_results['times'][dataset] = None
    
    return iter_results

def eval_all_iters(config):
    """Evaluate all checkpoints from min_iter to max_iter, skipping already evaluated iterations"""
    
    eval_config = config['eval_config']
    
    min_iter = eval_config['min_iter']
    max_iter = eval_config['max_iter']
    iter_step = eval_config['iter_step']
    datasets = eval_config['datasets']
    
    # Generate target iteration list
    target_iters = list(range(min_iter, max_iter + 1, iter_step))
    print(f"Target iterations: {len(target_iters)} iterations from {target_iters[0]} to {target_iters[-1]} (step: {iter_step})")
    
    # Load existing results
    existing_results = load_csv(config)
    
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
    
    # Results storage for new evaluations
    new_results = []
    
    # Evaluate each needed iteration
    for iter in tqdm(needed_iters, desc="Evaluating iterations"):
        try:
            # Check if we have partial results for this iteration
            iter_results = None
            for result in existing_results:
                if result['iter'] == iter:
                    iter_results = result
                    break
            
            # Get the specific datasets that need evaluation for this iteration
            needed_datasets = needed_datasets_per_iter.get(iter, datasets)
            
            iter_results = eval_one_iter_partial(iter, config, iter_results, needed_datasets, save_sol=False)
            new_results.append(iter_results)
                  
        except Exception as e:
            print(f"✗ Error evaluating iteration {iter}: {e}")
            # Add empty results for failed iteration
            new_results.append({
                'iter': iter,
                'scores': {dataset: None for dataset in datasets},
                'times': {dataset: None for dataset in datasets}
            })
    
    # Merge new results with existing results
    all_results = merge_results(existing_results, new_results)
    # save results
    score_df, time_df = save_results(all_results, config)
    
    return all_results

def results_to_df(results, eval_config):
    """Save evaluation results to solution_score and solution_time CSV files"""

    datasets = eval_config['datasets']
    
    # Prepare data for CSV
    iters = [result['iter'] for result in results]
    
    # Solution scores and times
    score_data = {'iter': iters}
    time_data = {'iter': iters}
    for dataset in datasets:
        scores = [result['scores'].get(dataset) for result in results]
        score_data[dataset] = scores
        times = [result['times'].get(dataset) for result in results]
        time_data[dataset] = times
    
    # Save to CSV
    score_df = pd.DataFrame(score_data)
    time_df = pd.DataFrame(time_data)
    return score_df,time_df


def save_results(results, config):
    """Save evaluation results to individual dataset CSV files
        results: list of dicts, each dict contains 'iter', 'scores', 'times'
        config: config file

        return: score_df, time_df , constructing from results
    """
    eval_config = config['eval_config']
    model_config = config['model_config']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    datasets = eval_config['datasets']

    # Create output directory if it doesn't exist
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    # Save individual dataset files
    for dataset in datasets:
        dataset_file = f"{save_result_dir}/{dataset}.csv"
        
        # Prepare data for this dataset
        dataset_data = []
        for result in results:
            if (result['scores'].get(dataset) is not None and 
                result['times'].get(dataset) is not None):
                dataset_data.append({
                    'iter': result['iter'],
                    'score': result['scores'][dataset],
                    'time': result['times'][dataset]
                })
        
        if dataset_data:
            # Create DataFrame and save
            df = pd.DataFrame(dataset_data)
            df.to_csv(dataset_file, index=False)
            print(f"  Saved {dataset}.csv with {len(df)} rows")
        else:
            print(f"  Warning: No valid data for {dataset}")
    
    print(f"Results saved to {save_result_dir}/")
    print(f"  - Individual dataset CSV files created")
    
    # Also create the combined DataFrames for backward compatibility
    score_df, time_df = results_to_df(results, eval_config)
    
    return score_df, time_df


def main():
    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # parse args
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--eval_all_iters", action="store_true")
    parser.add_argument("--eval_iter", type=int)
    parser.add_argument("--save_sol_only", action="store_true", help="save sol to <dataset>_iter.txt,not update csv")
    args = parser.parse_args()

    # eval_all_iters
    if args.eval_all_iters:
        print("\neval_all_iters is True, running comprehensive evaluation...")
        
        try:
            from testUtils import plot_val_eval_scores
            all_results = eval_all_iters(config)
            plot_val_eval_scores(config)
        except ImportError:
            print("Warning: eval_all_iterations.py not found, falling back to single iteration evaluation")
    
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

        from testUtils import eval_one_iter_partial,merge_results,save_results,get_evaluation_status,load_csv,load_sol
        target_iters = [args.eval_iter]
        eval_config = config['eval_config']
        datasets = eval_config['datasets']
        new_results = []
        if args.save_sol_only:
            # when save_sol, do not save results to prevent overwrite
            for iter in target_iters:
                existing_sols = load_sol(iter,config)
                evaled_datasets = [sol.split('/')[-1].split('_')[0] for sol in existing_sols]
                needed_datasets = [dataset for dataset in datasets if dataset not in evaled_datasets]
                print(f"Evaluating iteration {iter} with datasets: {needed_datasets}")
                eval_one_iter_partial(iter, config,specified_datasets = needed_datasets, save_sol=True)
            
        else:
            # similar to the logic of eval_all_iters, save sol and save results
            existing_results = load_csv(config)
            needed_iters, needed_datasets_per_iter = get_evaluation_status(existing_results, target_iters, datasets)
            if not needed_iters:
                print("No new evaluation needed, skipping evaluation and save_results...")
            else:
                new_results = []
                for iter in needed_iters:
                    iter_results = None
                    for result in existing_results:
                        if result['iter'] == iter:
                            iter_results = result
                            break
                    
                    # Get the specific datasets that need evaluation for this iteration
                    needed_datasets = needed_datasets_per_iter.get(iter, datasets)
                    new_iter_results = eval_one_iter_partial(iter, config, iter_results, needed_datasets, save_sol=True)
                    new_results.append(new_iter_results)
                # Merge new results with existing results
                all_results = merge_results(existing_results, new_results)
                save_results(all_results, config)


if __name__=="__main__":
    main()

# def GetSolution(dqn,stepRatio, datasets, dataset_dir, save_result_dir):
#     ######################################################################################################################
#     ##................................................Get Solution (model).....................................................
    
#     ## begin computing...
#     sol_time_df = pd.DataFrame(np.arange(1*len(datasets)).reshape((1,len(datasets))),index=['time'], columns=datasets)

#     for j in range(len(datasets)):
#         print ('\nTesting dataset %s'%datasets[j])
        
#         data_file = find_data_file(datasets[j], dataset_dir)
#         if data_file is None:
#             print(f"Warning: Could not find data file for {datasets[j]}")
#             continue

#         g_test = load_graph_from_file(data_file)
#         if g_test is None:
#             print(f"Warning: Could not load graph for {datasets[j]}")
#             continue
#         result_file = save_result_dir + datasets[j] + '.txt'
#         solution, time = dqn.EvaluateRealData(g_test, result_file, stepRatio)
#         sol_time_df.iloc[0,j] = time
#         print('Data:%s, get_sol_time:%.2f'%(datasets[j], time))
#     sol_time_df.to_csv(save_result_dir + 'solution_time.csv' , encoding='utf-8', index=False)

    

# def EvaluateSolution(dqn,strategyID,datasets, dataset_dir, save_result_dir ):
#     #######################################################################################################################
#     ##................................................Evaluate Solution.....................................................
   
#     ## begin computing...
#     score_df = pd.DataFrame(np.arange(1 * len(datasets)).reshape((1, len(datasets))), index=['solution'], columns=datasets)
#     for i in range(len(datasets)):
#         print('\nEvaluating dataset %s' % datasets[i])
#         data_file = find_data_file(datasets[i], dataset_dir)
#         if data_file is None:
#             print(f"Warning: Could not find data file for {datasets[i]}")
#             continue
#         g_test = load_graph_from_file(data_file)
#         if g_test is None:
#             print(f"Warning: Could not load graph for {datasets[i]}")
#             continue
#         solution = save_result_dir + datasets[i] + '.txt'
#         t1 = time.time()
#         # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
#         ################################## modify to choose which strategy to evaluate
#         score, MaxCCList = dqn.EvaluateSol(g_test, solution, strategyID, reInsertStep=0.001)
#         t2 = time.time()
#         print('Data: %s, score:%.6f, eval_sol_time: %.6f'% (datasets[i], score,t2 - t1))

#         score_df.iloc[0, i] = score
#         result_file = save_result_dir + 'MaxCCList_Strategy_' + datasets[i] + '.txt'
#         with open(result_file, 'w') as f_out:
#             for j in range(len(MaxCCList)):
#                 f_out.write('%.8f\n' % MaxCCList[j])
#     score_df.to_csv(save_result_dir + 'solution_score.csv', encoding='utf-8', index=False)

# def RandomRemoveEvaluate(dqn,STEPRATIO, REPEAT, MODEL_FILE_CKPT=None, datasets=None, dataset_dir=None, model_dir=None):

#     # randomRatioList = [0.005,0.01,0.02,0.05,0.1,0.15,0.2]
#     randomRatioList = [0.5]
#     ######################################################################################################################
#     ##................................................Get Solution (model).....................................................
#     # dqn = GraphDQN()
#     cfd = os.path.dirname(__file__)
#     ## data_set
#     data_test_path = dataset_dir or '%s/../../data/real/'%(cfd)
#     # datasets = ['Crime','HI-II-14']
#     ## model_file
#     model_file_path = model_dir or f'./models/'
#     if not MODEL_FILE_CKPT:
#         model_file_ckpt = dqn.findModel()
#     else:
#         model_file_ckpt = MODEL_FILE_CKPT
#     model_file = model_file_path + model_file_ckpt
#     ## modify to choose which stepRatio to get the solution
#     stepRatio = STEPRATIO
#     ## save_dir : save sol
#     save_dir = '%s/result/real'%(cfd)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     ## save_dir_local : save random test result
#     save_dir_local = save_dir + '/StepRatio_%.4f/evaluate_random' % stepRatio
#     if not os.path.exists(save_dir_local):
#         os.mkdir(save_dir_local)
        
    # ## begin computing...
    # print("------------------ random evaluate ------------------")
    # print('The best model is :%s'%(model_file))

    # for j in range(len(datasets)):
    #     print ('\nTesting dataset %s'%datasets[j])
    #     data_test = data_test_path + datasets[j] + '.txt'
    #     df = pd.DataFrame(np.arange(REPEAT*len(randomRatioList)).reshape((REPEAT,len(randomRatioList))), columns=randomRatioList)
    #     for r_i,r in enumerate(randomRatioList):
    #         for t in range(REPEAT):
    #             score, MaxCCList,solution, time = dqn.EvaluateRealData_random(model_file, data_test, save_dir, r, stepRatio)
    #             print("score:",score)
    #             # print("solution",solution)
    #             df.iloc[t,r_i] = score
    #     print('Data:%s, remove ratio:%f, time:%.2f'%(datasets[j], r, time))
    #     df.to_csv(save_dir_local + '/%s.csv'%(datasets[j]), encoding='utf-8', index=False)
