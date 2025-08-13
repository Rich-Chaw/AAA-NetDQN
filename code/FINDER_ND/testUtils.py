from calendar import c
from re import T
import sys,os

from tensorflow.python.keras.models import model_config
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
import math
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
    train_g_type=model_config['g_type']
    g_params = model_config['g_params']
    target_graph=model_config['target_graph']   
    save_model_dir=model_config['save_model_dir']
    
    dqn = GraphDQN(
        g_type=train_g_type,
        g_params  = g_params,
        gnn_model=gnn_model,
        target_graph=target_graph,
        save_model_dir=save_model_dir,
    )
    if iter is None:
        return dqn
    else:
        # e.g. GIN_iter_2700.ckpt
        ckpt_file = f"{gnn_model}_iter_{iter}.ckpt"
        dqn.LoadModel(ckpt_file)
        return dqn


def load_real_graph(dataset, dataset_dir):
    if dataset in ['Crime','HI-II-14','Digg','Enron','Gnutella31','Facebook','Epinions','Youtube','Flickr']:
        data_file = f'{dataset_dir}/%s.txt'%(dataset)
    elif dataset in ['corruption']:
        data_file = f'{dataset_dir}/%s.gt'%(dataset)
    else: 
        data_file = ''

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

def load_synthetic_graphs(g_type,**kwargs):
    graphs = []
    if g_type == "BA":
        g_num,nrange,m = kwargs.values()
        graphs_dir = f"../../dataset/synthetic/{g_type}/nrange_{nrange}/m_{m}"
        for i in range(g_num):
            g_path = f'{graphs_dir}/g_{i}'
            g = nx.read_gml(g_path,destringizer=int) # destringizer=int to convert label string to int
            graphs.append(g)
    else: pass
    return graphs



def eval_one_iter_partial(iter, config, iter_results=None, specified_datasets=None, save_sol=False):
    """Evaluate a single iteration checkpoint on specific datasets only"""
    print(f"\nEvaluating iteration {iter}...")
    
    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']

    # Reset TensorFlow graph/session before creating a new model
    try:
        import tensorflow as tf
        try:
            tf.compat.v1.reset_default_graph()
        except AttributeError:
            tf.keras.backend.clear_session()  # For TF 2.x
    except ImportError:
        pass  # If tensorflow is not available, skip

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
        return existing_results,False
    
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
    
    return all_results,True

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
    """Save evaluation results to solution_score and solution_time CSV fi les seperately
        results: list of dicts, each dict contains 'iter', 'scores', 'times'
        config: config file

        return: score_df, time_df , constructing from results
    """
    eval_config = config['eval_config']
    model_config = config['model_config']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')

    score_df, time_df = results_to_df(results, eval_config)
    
    score_file = f"{save_result_dir}/sol_score.csv"
    time_file = f"{save_result_dir}/sol_time.csv"

    score_df.to_csv(score_file, index=False)
    time_df.to_csv(time_file, index=False)
        
    print(f"Results saved to {save_result_dir}/")
    print(f"  - sol_score.csv")
    print(f"  - sol_time.csv")
    
    return score_df, time_df

def load_validation_scores(config):
    """Load validation scores from ModelVC CSV file"""
    model_config = config['model_config']
    # vc_file is generated when training
    vc_file = "%s/%s_nrange_%d_%d_m_%d/ModelVC_%s.csv"%(
                                                model_config['save_model_dir'],
                                                model_config['g_type'],
                                                model_config['g_params']['num_min'],
                                                model_config['g_params']['num_max'],
                                                model_config['g_params']['m'],
                                                model_config['gnn_model']
                                                )
    
    if not os.path.exists(vc_file):
        print(f"Warning: Validation file not found: {vc_file}")
        return None
    
    try:
        vc_df = pd.read_csv(vc_file, header=None)
        return vc_df
    except Exception as e:
        print(f"Error loading validation scores: {e}")
        return None

def plot_val_eval_scores(eval_df, val_df, config,min_iter = 0, max_iter = 30000,iter_step=300):
    """Create comparison plot of validation vs real dataset scores"""
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    
    if val_df is None:
        print("Warning: No validation scores available, skipping plot")
        return
    
    # Create figure
    row = math.ceil(math.sqrt(len(datasets)))
    col = row
    fig, axes = plt.subplots(row, col, figsize=(5*len(datasets), 5*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    s_i = min_iter // iter_step
    e_i = max_iter // iter_step
    val_iters = val_df.iloc[s_i:e_i, 0].values  # First column: iteration
    val_scores = val_df.iloc[s_i:e_i, 1].values  # Second column: validation score

    eval_iters = eval_df['iter'].values
    figname = "sol_score"
    for i, dataset in enumerate(datasets):
        figname += "_" + dataset  
        if len(axes) == 1:
            ax = axes[i]
        else:
            ax = axes[i//row][i%col]
        
        # Plot real dataset scores (emphasized)
        eval_scores = eval_df[dataset].values
        valid_mask = ~pd.isna(eval_scores)
        if np.any(valid_mask):
            ax.plot(eval_iters[valid_mask], eval_scores[valid_mask], 
                    'r-o', label=f'Eval {dataset}', markersize=2.0, linewidth=1.0, zorder=3)
        
        # Plot validation scores (de-emphasized)
        ax.plot(val_iters, val_scores, 
                color='blue', linestyle='--', marker='s', label='Valid', 
                markersize=1.5, linewidth=0.8, alpha=0.5, zorder=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Solution Score')
        ax.set_title(f'{dataset} Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_result_dir}/{figname}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_result_dir}/sol_score.png")
    plt.close()

def plot_eval_scores(csv_files, dataset, labels=None, title=None, save_dir=None, save_name=None, show=False, smooth_window=None):
    """Plot evaluation scores of different models on the same dataset.

    Parameters
    - csv_files: list of str
        Paths to sol_score.csv files for different models. Each CSV must have
        an 'iter' column and a column named after the target `dataset`.
    - dataset: str
        Dataset column to compare (e.g., '30-50', '50-100', 'Crime').
    - labels: list of str or None
        Optional display labels for each model; inferred from paths if None.
    - title: str or None
        Optional plot title. Defaults to f"Eval scores on {dataset}".
    - save_dir: str or None
        Directory to save the figure. If None, uses the directory of the first CSV.
    - save_name: str or None
        Filename (without directory) for the saved figure. If None, uses
        f"sol_score_compare_{dataset}.png".
    - show: bool
        If True, display the plot window (useful in notebooks). Defaults to False.
    - smooth_window: int or None
        Optional rolling window size for smoothing scores (moving average).

    Example
    >>> plot_eval_scores([
    ...   'AAA-NetDQN/code/result/temp/synthetic/barabasi_albert_nrange_30_50_m_4/graphSage_StepRatio_0.0100/sol_score.csv',
    ...   'AAA-NetDQN/code/result/temp/synthetic/barabasi_albert_nrange_30_50_m_1/graphSage_StepRatio_0.0100/sol_score.csv'
    ... ], dataset='30-50')
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    if not csv_files:
        print("plot_eval_scores: no csv_files provided")
        return

    # Infer labels from path if not provided
    if labels is None:
        inferred_labels = []
        for path in csv_files:
            # Use the parent directory name (often contains model name and StepRatio)
            try:
                parent = os.path.basename(os.path.dirname(path))
                inferred_labels.append(parent if parent else os.path.basename(path))
            except Exception:
                inferred_labels.append(os.path.basename(path))
        labels = inferred_labels

    if len(labels) != len(csv_files):
        raise ValueError("labels must be the same length as csv_files")

    # Prepare output path
    if save_dir is None:
        save_dir = os.path.dirname(csv_files[0]) if os.path.dirname(csv_files[0]) else "."
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if save_name is None:
        safe_dataset = str(dataset).replace('/', '_')
        save_name = f"sol_score_compare_{safe_dataset}.png"
    save_path = os.path.join(save_dir, save_name)

    # Plot
    plt.figure(figsize=(8, 5))
    plotted_any = False

    for path, label in zip(csv_files, labels):
        if not os.path.exists(path):
            print(f"Warning: file not found, skip: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            continue

        if 'iter' not in df.columns:
            print(f"Warning: 'iter' column missing in {path}, skip")
            continue
        if dataset not in df.columns:
            print(f"Warning: dataset '{dataset}' not found in {path}, available: {list(df.columns)}")
            continue

        # Clean and sort
        sub = df[['iter', dataset]].copy()
        # Coerce to numeric and drop NaNs
        sub['iter'] = pd.to_numeric(sub['iter'], errors='coerce')
        sub[dataset] = pd.to_numeric(sub[dataset], errors='coerce')
        sub = sub.dropna(subset=['iter', dataset]).sort_values('iter')
        if sub.empty:
            print(f"Warning: no valid rows after cleaning for {path}")
            continue

        x = sub['iter'].values
        y = sub[dataset].values

        if smooth_window is not None and isinstance(smooth_window, int) and smooth_window > 1:
            try:
                y_series = pd.Series(y).rolling(window=smooth_window, min_periods=max(1, smooth_window // 2)).mean()
                y = y_series.values
            except Exception:
                pass

        plt.plot(x, y, marker='o', markersize=2.0, linewidth=1.2, label=label)
        plotted_any = True

    if not plotted_any:
        print("plot_eval_scores: nothing to plot after processing inputs")
        plt.close()
        return

    plt.xlabel('Iteration')
    plt.ylabel('Solution Score')
    plt.title(title if title is not None else f"Eval scores on {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()

def detailed_result_dir(model_config,eval_config,mode='real'):

    save_result_dir = eval_config['save_result_dir']
    # real/barabasi_albert_nrange_30_50_m_4/GIN
    sub_dir = '/%s/%s_nrange_%d_%d_m_%d/%s_StepRatio_%.4f/' %(mode, 
                                                model_config['g_type'],
                                                model_config['g_params']['num_min'],
                                                model_config['g_params']['num_max'],
                                                model_config['g_params']['m'],
                                                model_config['gnn_model'],
                                                eval_config['step_ratio'])
    save_result_dir = save_result_dir + sub_dir

    # Create save directory
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    return save_result_dir

def load_csv(config):
    """Load existing evaluation results from CSV files
        for datasets in eval_config
    """
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    
    score_file = f"{save_result_dir}/sol_score.csv"
    time_file = f"{save_result_dir}/sol_time.csv"
    
    print(f"Looking for existing results in: {save_result_dir}")
    print(f"Score file: {score_file}")
    print(f"Time file: {time_file}")
    
    existing_results = []
    
    if os.path.exists(score_file) and os.path.exists(time_file):
        try:
            score_df = pd.read_csv(score_file)
            time_df = pd.read_csv(time_file)
            
            print(f"Found existing files with {len(score_df)} rows")
            print(f"Score columns: {list(score_df.columns)}")
            print(f"Time columns: {list(time_df.columns)}")
            
            # Convert DataFrames back to results format
            for idx, row in score_df.iterrows():
                iter_num = int(row['iter'])
                scores = {}
                times = {}
                
                for dataset in datasets:
                    if dataset in row and pd.notna(row[dataset]):
                        scores[dataset] = float(row[dataset])
                    else:
                        scores[dataset] = None
                    
                    if dataset in time_df.columns and pd.notna(time_df.iloc[idx][dataset]):
                        times[dataset] = float(time_df.iloc[idx][dataset])
                    else:
                        times[dataset] = None
                
                existing_results.append({
                    'iter': iter_num,
                    'scores': scores,
                    'times': times
                })
            
            print(f"✓ Loaded {len(existing_results)} existing evaluation results")
            if existing_results:
                print(f"  Iterations found: {[r['iter'] for r in existing_results]}")
            
        except Exception as e:
            print(f"✗ Warning: Error loading existing results: {e}")
            existing_results = []
    else:
        print("✗ No existing results found, starting fresh evaluation")
        if not os.path.exists(score_file):
            print(f"  Score file not found: {score_file}")
        if not os.path.exists(time_file):
            print(f"  Time file not found: {time_file}")
    
    return existing_results


def load_sol(iter,config):
    """load solution files for a given iteration"""
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    sol_files = []
    for dataset in datasets:
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

def test_evaluation_logic(config):
    """Test the evaluation logic without running actual evaluation"""
    print("\n" + "="*60)
    print("TESTING EVALUATION LOGIC")
    print("="*60)
    
    eval_config = config['eval_config']
    datasets = eval_config['datasets']
    min_iter = eval_config['min_iter']
    max_iter = eval_config['max_iter']
    iter_step = eval_config['iter_step']
    
    target_iters = list(range(min_iter, max_iter + 1, iter_step))
    
    print(f"Target iterations: {target_iters}")
    print(f"Datasets: {datasets}")
    
    # Load existing results
    existing_results = load_csv(config)
    
    # Determine what needs to be evaluated
    needed_iters, needed_datasets_per_iter = get_evaluation_status(existing_results, target_iters, datasets)
    
    # Print evaluation summary
    print_evaluation_summary(existing_results, target_iters, datasets)
    
    # Print evaluation plan
    print_evaluation_plan(needed_iters, needed_datasets_per_iter, datasets)
    
    print("="*60)
    return needed_iters, needed_datasets_per_iter



