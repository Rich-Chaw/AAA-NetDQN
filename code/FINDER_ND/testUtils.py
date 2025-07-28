import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import json
import matplotlib.pyplot as plt

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_model(model_config,iter = None):
    """Create GraphDQN model from configuration"""
    gnn_model = model_config['gnn_model']
    num_min = model_config['num_min']
    num_max = model_config['num_max']
    g_type=model_config['g_type']
    target_graph=model_config['target_graph']   
    save_model_dir=model_config['save_model_dir']
    if iter is None:
        # dqn will find the best iter and load
        ckpt_file = None
    else:
        # e.g. GIN_nrange_30_50_iter_2700.ckpt
        ckpt_file = f"{gnn_model}_nrange_{num_min}_{num_max}_iter_{iter}.ckpt"
    
    dqn = GraphDQN(
        g_type=g_type,
        gnn_model=gnn_model,
        target_graph=target_graph,
        num_min=num_min,
        num_max=num_max,
        save_model_dir=save_model_dir,
        ckpt_file = ckpt_file
    )
    return dqn

def find_data_file(dataset_name, dataset_dir):
    if dataset_name in ['Crime','HI-II-14','Digg','Enron','Gnutella31','Facebook','Epinions','Youtube','Flickr']:
        data_file = f'{dataset_dir}/%s.txt'%(dataset_name)
    elif dataset_name in ['corruption']:
        data_file = f'{dataset_dir}/%s.gt'%(dataset_name)
    else: 
        data_file = None
    
    return data_file


def load_graph_from_file(data_file):
    if data_file.split('.')[-1] == 'txt': 
        G = nx.read_edgelist(data_file,nodetype=int)
    else:
        try:
            import graph_tool.all as gt
            gt_graph = gt.load_graph(data_file)
            G = nx.Graph()
            for e in gt_graph.edges():
                u = int(e.source())
                v = int(e.target())
                G.add_edge(u, v)
        except ImportError:
            print("Warning: graph_tool not available, skipping .gt files")
            return None
    return G

def eval_one_iter(iter, config):
    """Evaluate a single iteration checkpoint on all datasets"""
    print(f"\nEvaluating iteration {iter}...")
    
    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']

    # Create model for this iteration
    dqn = create_model(model_config, iter)
    
    datasets = eval_config['datasets']
    step_ratio = eval_config['step_ratio']
    strategy_id = eval_config['strategy_id']
    dataset_dir = data_config['dataset_dir']
    
    # Results for this iteration
    iter_results = {
        'iter': iter,
        'scores': {},
        'times': {}
    }
    
    # Evaluate each dataset
    for dataset in datasets:
        print(f"  Evaluating dataset: {dataset}")
        
        # Find and load graph
        data_file = find_data_file(dataset, dataset_dir)
        if data_file is None:
            print(f"    Warning: Could not find data file for {dataset}")
            iter_results['scores'][dataset] = None
            iter_results['times'][dataset] = None
            continue
            
        g_test = load_graph_from_file(data_file)
        if g_test is None:
            print(f"    Warning: Could not load graph for {dataset}")
            iter_results['scores'][dataset] = None
            iter_results['times'][dataset] = None
            continue
        
        # Get solution
        temp_result_file = f"temp_{dataset}_{iter}.txt"
        try:
            solution, get_time = dqn.EvaluateRealData(g_test, temp_result_file, step_ratio)
            
            # Evaluate solution
            t1 = time.time()
            score, MaxCCList = dqn.EvaluateSol(g_test, temp_result_file, strategy_id, reInsertStep=0.001)
            eval_time = time.time() - t1
            
            iter_results['scores'][dataset] = score
            iter_results['times'][dataset] = get_time + eval_time
            
            print(f"    Score: {score:.6f}, Total time: {get_time + eval_time:.2f}s")
            
            # Clean up temp file
            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
                
        except Exception as e:
            print(f"    Error evaluating {dataset}: {e}")
            iter_results['scores'][dataset] = None
            iter_results['times'][dataset] = None
    
    return iter_results

def eval_all_iters(config):
    """Evaluate all checkpoints from 0 to max_iter"""
    
    eval_config = config['eval_config']
    
    max_iter = eval_config['max_iter']
    iter_step = eval_config['iter_step']
    datasets = eval_config['datasets']
    
    # Generate iteration list
    iters = list(range(0, max_iter + 1, iter_step))
    print(f"Evaluating {len(iters)} iterations: {iters[0]} to {iters[-1]} (step: {iter_step})")
    
    # Results storage
    all_iter_results = []
    
    # Evaluate each iteration
    for iter_num in tqdm(iters, desc="Evaluating iterations"):
        try:
            iter_results = eval_one_iter(iter_num, config)
            all_iter_results.append(iter_results)
        except Exception as e:
            print(f"Error evaluating iteration {iter_num}: {e}")
            # Add empty results for failed iteration
            all_iter_results.append({
                'iter': iter_num,
                'scores': {dataset: None for dataset in datasets},
                'times': {dataset: None for dataset in datasets}
            })
    
    return all_iter_results

def save_results(results, config):
    """Save evaluation results to solution_score and solution_time CSV files seperately
        results: list of dicts, each dict contains 'iter', 'scores', 'times'
    
    """
    eval_config = config['eval_config']
    datasets = eval_config['datasets']
    save_result_dir = eval_config['save_result_dir']
    
    # Create save directory
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    # Prepare data for CSV
    iters = [result['iter'] for result in results]
    
    # Solution scores
    score_data = {'iter': iters}
    for dataset in datasets:
        scores = [result['scores'].get(dataset) for result in results]
        score_data[dataset] = scores
    
    # Solution times
    time_data = {'iter': iters}
    for dataset in datasets:
        times = [result['times'].get(dataset) for result in results]
        time_data[dataset] = times
    
    # Save to CSV
    score_df = pd.DataFrame(score_data)
    time_df = pd.DataFrame(time_data)
    
    score_df.to_csv(f"{save_result_dir}/solution_score.csv", index=False)
    time_df.to_csv(f"{save_result_dir}/solution_time.csv", index=False)
    
    print(f"Results saved to {save_result_dir}/")
    print(f"  - solution_score.csv")
    print(f"  - solution_time.csv")
    
    return score_df, time_df

def load_validation_scores(config):
    """Load validation scores from ModelVC CSV file"""
    model_config = config['model_config']

    gnn_model = model_config['gnn_model']
    g_type = model_config['g_type']
    num_min = model_config['num_min']
    num_max = model_config['num_max']
    model_dir = model_config['save_model_dir']
    model_dir = model_dir + f'/{g_type}'
    
    vc_file = f"{model_dir}/ModelVC_{gnn_model}_{num_min}_{num_max}.csv"
    
    if not os.path.exists(vc_file):
        print(f"Warning: Validation file not found: {vc_file}")
        return None
    
    try:
        vc_data = pd.read_csv(vc_file, header=None)
        # Extract iteration numbers and validation scores
        iters = vc_data.iloc[:, 0].values  # First column: iteration
        val_scores = vc_data.iloc[:, 1].values  # Second column: validation score
        return dict(zip(iters, val_scores))
    except Exception as e:
        print(f"Error loading validation scores: {e}")
        return None

def create_comparison_plot(score_df, val_scores, config):
    """Create comparison plot of validation vs real dataset scores"""
    eval_config = config['eval_config']
    datasets = eval_config['datasets']
    save_result_dir = eval_config['save_result_dir']
    
    if val_scores is None:
        print("Warning: No validation scores available, skipping plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    iters = score_df['iter'].values
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Plot real dataset scores
        real_scores = score_df[dataset].values
        valid_mask = ~pd.isna(real_scores)
        if np.any(valid_mask):
            ax.plot(iters[valid_mask], real_scores[valid_mask], 
                   'b-o', label=f'{dataset} (Real)', markersize=3)
        
        # Plot validation scores
        val_iters = list(val_scores.keys())
        val_scores_list = list(val_scores.values())
        if val_iters:
            ax.plot(val_iters, val_scores_list, 
                   'r-s', label='Validation', markersize=3)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Solution Score')
        ax.set_title(f'{dataset} Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_result_dir}/sol_score.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_result_dir}/sol_score.png")
    plt.close()