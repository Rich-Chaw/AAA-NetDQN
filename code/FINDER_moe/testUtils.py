from calendar import c
from re import T
import sys,os
import itertools

from tensorflow.python.keras.models import model_config
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
from MoEGraphDQN import MoEGraphDQN
import numpy as np
import math
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def load_config(config_file=f'{os.path.dirname(__file__)}/config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_moe_model(model_config,iter = None):
    """Create MoEGraphDQN model from configuration"""
    gnn_model = model_config['gnn_model']
    train_g_type=model_config['g_type']
    g_params = model_config['g_params']
    target_graph=model_config['target_graph']   
    save_model_dir=model_config['save_model_dir']

    # Reset TensorFlow graph/session before creating a new model
    try:
        import tensorflow as tf
        try:
            tf.compat.v1.reset_default_graph()
        except AttributeError:
            tf.keras.backend.clear_session()  # For TF 2.x
    except ImportError:
        pass  # If tensorflow is not available, skip
    
    moe_config = {
        'num_experts': 4,
        'top_k': 2,
        'router_dropout': 0.1,
        'load_balance_loss_weight': 0.01
    }
    
    moe_dqn = MoEGraphDQN(
        g_type = train_g_type,
        g_params = g_params,
        target_graph = target_graph,
        save_model_dir= save_model_dir,
        moe_config = moe_config
    )

    if iter is None:
        return moe_dqn
    else:
        # e.g. GIN_iter_2700.ckpt
        ckpt_file = f"{gnn_model}_iter_{iter}.ckpt"
        moe_dqn.LoadModel(ckpt_file)
        return moe_dqn

def create_model(model_config,iter = None):
    """Create GraphDQN model from configuration"""
    gnn_model = model_config['gnn_model']
    train_g_type=model_config['g_type']
    g_params = model_config['g_params']
    target_graph=model_config['target_graph']   
    save_model_dir=model_config['save_model_dir']

    # Reset TensorFlow graph/session before creating a new model
    try:
        import tensorflow as tf
        try:
            tf.compat.v1.reset_default_graph()
        except AttributeError:
            tf.keras.backend.clear_session()  # For TF 2.x
    except ImportError:
        pass  # If tensorflow is not available, skip
    

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


def _get_synth_param_grid(eval_config):
    """Build parameter grid for synthetic datasets from eval_config.
    Supports families like BA (nranges, m_values) and small-world (nranges, k_values, p_values).
    Returns a list of dicts with keys 'synth_dataset', 'filename', and 'params'.
    """
    g_type = eval_config['synthetic_g_type']
    params_cfg = eval_config.get('synthetic_g_params', {})
    per_graphs = eval_config['per_graphs']

    grid = []
    if g_type == 'BA':
        nranges = params_cfg.get('nranges', [])
        m_values = params_cfg.get('m_values', [])
        for nrange, m in itertools.product(nranges, m_values):
            synth_dataset = f"{g_type}_nrange_{nrange}_m_{m}"
            filename = f"{g_type}_nrange_{nrange}_m_{m}_gn_{per_graphs}.csv"
            grid.append({
                'synth_dataset': synth_dataset,
                'filename': filename,
                'params': {'nrange': nrange, 'm': m}
            })
    elif g_type == 'PL':
        # Future families, e.g., small-world: nranges, k_values, p_values
        nranges = params_cfg.get('nranges', [])
        k_values = params_cfg.get('k_values', [])
        p_values = params_cfg.get('p_values', [])
        for nrange, k, p in itertools.product(nranges, k_values, p_values):
            # Sanitize p for filename
            p_str = str(p).replace('.', '_')
            synth_dataset = f"{g_type}_nrange_{nrange}_k_{k}_p_{p_str}"
            filename = f"{g_type}_nrange_{nrange}_k_{k}_p_{p_str}_gn_{per_graphs}.csv"
            grid.append({
                'synth_dataset': synth_dataset,
                'filename': filename,
                'params': {'nrange': nrange, 'k': k, 'p': p}
            })
    return grid

# ------------------------------------------load graphs-------------------------------------------------------------------------
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

# ---------------------------------------------load csv------------------------------------------------------------
def load_real_csv(model_config,eval_config):
    """Load existing evaluation results from individual dataset CSV files

        Returns:
        - existing_results: dict
            {dataset_name: {iter_num: {'score': score, 'time': time}}}
    """

    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    
    print(f"Looking for existing real dataset results in: {save_result_dir}")
    existing_results = {} # New format: {dataset_name: {iter_num: {'score': score, 'time': time}}}
    for dataset in datasets:
        dataset_file = f"{save_result_dir}/{dataset}.csv"
        
        if os.path.exists(dataset_file):
            try:
                df = pd.read_csv(dataset_file)
                if not all(col in df.columns for col in ['iter', 'score', 'time']):
                    print(f"  Warning: {dataset}.csv missing required columns. Found: {list(df.columns)}")
                    continue
                
                df_clean = df.dropna(subset=['iter', 'score', 'time'])
                if df_clean.empty:
                    print(f"  Warning: {dataset}.csv has no valid data after cleaning")
                    continue
                
                existing_results[dataset] = {}
                for _, row in df_clean.iterrows():
                    iter_num = int(row['iter'])
                    existing_results[dataset][iter_num] = {
                        'score': float(row['score']),
                        'time': float(row['time'])
                    }
                print(f"  ✓ Loaded {dataset}.csv with {len(df_clean)} valid rows for {len(existing_results[dataset])} iterations")
                
            except Exception as e:
                print(f"  ⚠ Error loading {dataset}.csv: {e}")
        else:
            print(f"  - No existing results for dataset: {dataset} (file not found: {dataset_file})")
            
    if existing_results:
        total_iters_loaded = sum(len(iters) for iters in existing_results.values())
        print(f"✓ Successfully loaded existing results for {len(existing_results)} datasets, totaling {total_iters_loaded} iteration-dataset combinations.")
    else:
        print("✗ No existing real dataset results found to load.")
            
    return existing_results

def load_synth_csv(model_config,eval_config):
    """Load existing evaluation results from individual dataset CSV files

        Returns:
        - existing_results: dict
            {dataset_name: {iter_num: {'score': score, 'time': time}}}
    """

    synth_grid = _get_synth_param_grid(eval_config)
    synth_datasets = [g['synth_dataset'] for g in synth_grid]
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='synthetic')
    
    print(f"Looking for existing synthetic dataset results in: {save_result_dir}")
    existing_results = {} # New format: {dataset_name: {iter_num: {'score': score, 'time': time}}}
    for dataset in synth_datasets:
        dataset_file = f"{save_result_dir}/{dataset}_gn_{eval_config['per_graphs']}.csv"
        if os.path.exists(dataset_file):
            try:
                df = pd.read_csv(dataset_file)
                
                if not all(col in df.columns for col in ['iter', 'score', 'time']):
                    print(f"  Warning: {dataset}.csv missing required columns. Found: {list(df.columns)}")
                    continue
                
                df_clean = df.dropna(subset=['iter', 'score', 'time'])
                if df_clean.empty:
                    print(f"  Warning: {dataset}.csv has no valid data after cleaning")
                    continue
                
                existing_results[dataset] = {}
                for _, row in df_clean.iterrows():
                    iter_num = int(row['iter'])
                    existing_results[dataset][iter_num] = {
                        'score': float(row['score']),
                        'time': float(row['time'])
                    }
                print(f"  ✓ Loaded {dataset}.csv with {len(df_clean)} valid rows for {len(existing_results[dataset])} iterations")
                
            except Exception as e:
                print(f"  ⚠ Error loading {dataset}.csv: {e}")
        else:
            print(f"  - No existing results for dataset: {dataset} (file not found: {dataset_file})")
            
    if existing_results:
        total_iters_loaded = sum(len(iters) for iters in existing_results.values())
        print(f"✓ Successfully loaded existing results for {len(existing_results)} datasets, totaling {total_iters_loaded} iteration-dataset combinations.")
    else:
        print("✗ No existing synthetic dataset results found to load.")
            
    return existing_results

def load_validation_scores(config):
    """Load validation scores from ModelVC CSV file"""
    model_config = config['model_config']
    # vc_file is generated when training
    vc_file = "%s/%s_nrange_%s_m_%d/ModelVC_%s.csv"%(
                                                model_config['save_model_dir'],
                                                model_config['g_type'],
                                                model_config['g_params']['nrange'],
                                                model_config['g_params']['m'],
                                                model_config['gnn_model']
                                                )
    
    if not os.path.exists(vc_file):
        print(f"Warning: Validation file not found: {vc_file}")
        return None
    
    try:
        vc_df = pd.read_csv(vc_file, names=['iter','score','time_all'])
        return vc_df
    except Exception as e:
        print(f"Error loading validation scores: {e}")
        return None

def plot_val_eval_scores(config,min_iter = 0, max_iter = 30000,iter_step=300):
    """Create comparison plot of validation vs real dataset scores"""
    eval_config = config['eval_config']
    model_config = config['model_config']
    datasets = eval_config['datasets']
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    
  
    # Create figure
    row = math.ceil(math.sqrt(len(datasets)))
    col = row
    fig, axes = plt.subplots(row, col, figsize=(5*len(datasets), 5*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    figname = "sol_score"
    for i, dataset in enumerate(datasets):
        figname += "_" + dataset  
        if len(axes) == 1:
            ax = axes[i]
        else:
            ax = axes[i//row][i%col]
        
        # Load dataset-specific data from individual CSV file
        dataset_file = f"{save_result_dir}/{dataset}.csv"
        if os.path.exists(dataset_file):
            try:
                dataset_df = pd.read_csv(dataset_file)
                if 'iter' in dataset_df.columns and 'score' in dataset_df.columns:
                    # Clean and sort data
                    clean_df = dataset_df.dropna(subset=['iter', 'score'])
                    if not clean_df.empty:
                        clean_df = clean_df.sort_values('iter')
                        eval_iters = clean_df['iter'].values
                        eval_scores = clean_df['score'].values
                        
                        # Plot real dataset scores (emphasized)
                        valid_mask = ~pd.isna(eval_scores)
                        if np.any(valid_mask):
                            ax.plot(eval_iters[valid_mask], eval_scores[valid_mask], 
                                    'r-o', label=f'Eval {dataset}', markersize=2.0, linewidth=1.0, zorder=3)
                else:
                    print(f"Warning: {dataset}.csv missing required columns")
            except Exception as e:
                print(f"Warning: Error loading {dataset}.csv: {e}")
        else:
            print(f"Warning: Dataset file not found: {dataset_file}")
        
        val_df = load_validation_scores(config)
        val_df = val_df[val_df['iter'].isin(eval_iters)]
        val_iters = val_df['iter'].values  # First column: iteration
        val_scores = val_df['score'].values  # Second column: validation score
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
    print(f"Plot saved to {save_result_dir}/{figname}.png")
    plt.close()

def plot_eval_scores(csv_files, dataset, labels=None, title=None, save_dir=None, save_name=None, show=False, smooth_window=None):
    """Plot evaluation scores of different models on the same dataset.

    Parameters
    - csv_files: list of str
        Paths to individual dataset CSV files for different models. Each CSV must have
        columns 'iter', 'score', 'time' and contain data for the target `dataset`.
        For the new format, use paths like:
        'AAA-NetDQN/code/result/temp/synthetic/BA_nrange_30_50_m_4/graphSage_StepRatio_0.0100/30_50.csv'
    - dataset: str
        Dataset name to compare (e.g., 'Crime').
        This should match the filename (without .csv extension) of the CSV files.
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
    ...   'AAA-NetDQN/code/result/temp/synthetic/BA_nrange_30_50_m_4/graphSage_StepRatio_0.0100/Crime.csv',
    ...   'AAA-NetDQN/code/result/temp/synthetic/BA_nrange_30_50_m_1/graphSage_StepRatio_0.0100/Crime.csv'
    ... ], dataset='Crime')
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
        
        # For new format: check if 'score' column exists (instead of dataset-specific column)
        if 'score' not in df.columns:
            print(f"Warning: 'score' column missing in {path}, available: {list(df.columns)}")
            continue

        # Clean and sort - use 'score' column directly
        sub = df[['iter', 'score']].copy()
        # Coerce to numeric and drop NaNs
        sub['iter'] = pd.to_numeric(sub['iter'], errors='coerce')
        sub['score'] = pd.to_numeric(sub['score'], errors='coerce')
        sub = sub.dropna(subset=['iter', 'score']).sort_values('iter')
        if sub.empty:
            print(f"Warning: no valid rows after cleaning for {path}")
            continue

        x = sub['iter'].values
        y = sub['score'].values # Use 'score' column directly

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
    '''
        eval_config['save_result_dir']: e.g. './result'

        return detailed result dir e.g. './result/real/barabasi_albert_nrange_30_50_m_4/GIN'
    '''
    save_result_dir = eval_config['save_result_dir']
    # model/train_dataset/gnn_step
    # real/barabasi_albert_nrange_30_50_m_4/GIN_StepRatio_0.0100
    sub_dir = '/%s/%s_nrange_%s_m_%d/%s_StepRatio_%.4f/' %(mode, 
                                                model_config['g_type'],
                                                model_config['g_params']['nrange'],
                                                model_config['g_params']['m'],
                                                model_config['gnn_model'],
                                                eval_config['step_ratio'])
    save_result_dir = save_result_dir + sub_dir

    # Create save directory
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    return save_result_dir

def visualize_graph_dismantling(g, sol, viz_file_path):
    """
    Visualize the graph dismantling process step by step
    
    Parameters:
    - g: networkx graph
    - sol: list of nodes to remove in order
    """
    print(f"\n Creating dismantling visualization...")
    
    # Input validation
    if not sol or len(sol) == 0:
        print(f"  ⚠ Empty solution, skipping visualization")
        return
    # Check if all solution nodes exist in the graph
    check_sol_in_graph(g, sol)
    
    graph_id = int(viz_file_path.split("g_")[-1].split("_")[0])
    iter_num = int(viz_file_path.split("iter_")[-1].split("_")[0])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Graph Dismantling Process - Graph {graph_id}, Iter {iter_num}\n'
                 f'Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}', 
                 fontsize=16, fontweight='bold')
    
    # Original graph
    ax = axes[0, 0]
    pos = nx.spring_layout(g, seed=42)
    nx.draw(g, pos, ax=ax, node_color='lightblue', node_size=300, 
            edge_color='gray', width=1, with_labels=True, font_size=8)
    ax.set_title('Original Graph', fontweight='bold')
    ax.text(0.02, 0.98, f'Nodes: {g.number_of_nodes()}\nEdges: {g.number_of_edges()}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Step-by-step dismantling (3 key steps)
    dismantling_steps = [len(sol)//4, len(sol)//2, len(sol)]
    colors = ['orange', 'red', 'darkred']
    
    for idx, step in enumerate(dismantling_steps):
        if step == 0 or step > len(sol):
            continue
            
        ax = axes[0, idx+1] if idx < 2 else axes[1, 0]
        
        # Create a copy of the graph
        g_temp = g.copy()
        
        # Remove nodes up to this step
        nodes_to_remove = sol[:step]
        g_temp.remove_nodes_from(nodes_to_remove)
        
        # Get remaining components
        components = list(nx.connected_components(g_temp))
        largest_cc = max(components, key=len) if components else set()
        
        # Color nodes based on their status
        node_colors = []
        for node in g.nodes():
            if node in nodes_to_remove:
                node_colors.append('red')  # Removed nodes
            elif node in largest_cc:
                node_colors.append('lightgreen')  # Largest component
            else:
                node_colors.append('lightgray')  # Other components
        
        # Draw the original graph with updated colors
        nx.draw(g, pos, ax=ax, node_color=node_colors, node_size=300,
                edge_color='gray', width=1, with_labels=True, font_size=8)
        
        # Highlight removed nodes with red outline
        nx.draw_networkx_nodes(g, pos, nodelist=nodes_to_remove, 
                              node_color='red', node_size=300, ax=ax)
        
        step_title = f'After removing {step} nodes\nLargest CC: {len(largest_cc)}'
        ax.set_title(step_title, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightgreen', label=f'Largest CC ({len(largest_cc)})'),
            mpatches.Patch(color='red', label=f'Removed ({step})'),
            mpatches.Patch(color='lightgray', label='Other components')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Final dismantled state
    ax = axes[1, 1]
    g_final = g.copy()
    g_final.remove_nodes_from(sol)
    components_final = list(nx.connected_components(g_final))
    
    if components_final:
        largest_cc_final = max(components_final, key=len)
        pos_final = {node: pos[node] for node in g_final.nodes()}
        
        nx.draw(g_final, pos_final, ax=ax, node_color='lightgreen', 
                node_size=300, edge_color='gray', width=1, 
                with_labels=True, font_size=8)
        ax.set_title(f'Final State\nLargest CC: {len(largest_cc_final)}', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'All nodes removed\nGraph dismantled', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, fontweight='bold')
        ax.set_title('Final State - Graph Dismantled', fontweight='bold')
    
    # Component size distribution
    ax = axes[1, 2]
    if components_final:
        component_sizes = [len(comp) for comp in components_final]
        ax.hist(component_sizes, bins=min(20, len(component_sizes)), 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Component Size')
        ax.set_ylabel('Frequency')
        ax.set_title('Component Size Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No components\nremaining', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, fontweight='bold')
        ax.set_title('Component Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization

    plt.savefig(viz_file_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved dismantling visualization: {viz_file_path}")
    
    
    plt.close()

def create_dismantling_animation(g, sol, anim_file_path):
    """
    Create an animated visualization of the dismantling process
    
    Parameters:
    - g: networkx graph
    - sol: list of nodes to remove in order
    """
    print(f"  Creating dismantling animation...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(g, seed=42)
    
    def animate(frame):
        ax.clear()
        
        # Calculate how many nodes to remove at this frame
        nodes_removed = int((frame / 100) * len(sol))
        nodes_to_remove = sol[:nodes_removed]
        
        # Create temporary graph
        g_temp = g.copy()
        g_temp.remove_nodes_from(nodes_to_remove)
        
        # Get components
        components = list(nx.connected_components(g_temp))
        largest_cc = max(components, key=len) if components else set()
        
        # Color nodes
        node_colors = []
        for node in g.nodes():
            if node in nodes_to_remove:
                node_colors.append('red')
            elif node in largest_cc:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Draw graph
        nx.draw(g, pos, ax=ax, node_color=node_colors, node_size=300,
                edge_color='gray', width=1, with_labels=True, font_size=8)
        
        # Highlight removed nodes
        if nodes_to_remove:
            nx.draw_networkx_nodes(g, pos, nodelist=nodes_to_remove, 
                                  node_color='red', node_size=300, ax=ax)
        
        # Update title
        progress = (frame / 100) * 100
        ax.set_title(f'Graph Dismantling Progress: {progress:.1f}%\n'
                     f'Nodes removed: {nodes_removed}/{len(sol)}\n'
                     f'Largest CC: {len(largest_cc)}', 
                     fontweight='bold', fontsize=12)
        
        # Add progress bar
        progress_bar = FancyBboxPatch((0.1, 0.02), progress/100 * 0.8, 0.02, 
                                     boxstyle="round,pad=0.01", 
                                     facecolor='blue', alpha=0.7)
        ax.add_patch(progress_bar)
        
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=101, 
                                  interval=100, blit=False, repeat=False)
    
    # Save animation
    anim.save(anim_file_path, writer='pillow', fps=10)
    print(f"  ✓ Saved dismantling animation: {anim_file_path}")
    
    plt.close()

def plot_max_cc_curve(MaxCCList, plot_file_path):
    """
    Plot the MaxCC curve showing how the largest connected component size changes
    
    Parameters:
    - MaxCCList: list of largest connected component sizes
    - save_dir: directory to save the plot
    - graph_id: identifier for the graph
    - iter_num: iteration number
    """
    print(f"  Creating MaxCC curve plot...")
    
    # Input validation
    if not MaxCCList or len(MaxCCList) == 0:
        print(f"  ⚠ Empty MaxCCList, skipping plot")
        return
    
    graph_id = int(plot_file_path.split("g_")[-1].split("_")[0])
    iter_num = int(plot_file_path.split("iter_")[-1].split("_")[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert to numpy array and normalize
    max_cc_array = np.array(MaxCCList)
    nodes_removed = np.arange(len(max_cc_array))
    
    # Plot 1: MaxCC vs Nodes Removed
    ax1.plot(nodes_removed, max_cc_array, 'b-o', linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel('Number of Nodes Removed', fontweight='bold')
    ax1.set_ylabel('Largest Connected Component Size', fontweight='bold')
    ax1.set_title(f'MaxCC Curve - Graph {graph_id}, Iter {iter_num}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(max_cc_array))
    ax1.set_ylim(0, max_cc_array[0] * 1.05)
    
    # Add annotations for key points
    # Find where MaxCC drops significantly
    if len(max_cc_array) > 1:
        # Find the first significant drop (e.g., 50% of original)
        threshold = max_cc_array[0] * 0.5
        significant_drop_idx = np.where(max_cc_array <= threshold)[0]
        if len(significant_drop_idx) > 0:
            first_drop = significant_drop_idx[0]
            ax1.annotate(f'50% drop at {first_drop} nodes', 
                        xy=(first_drop, max_cc_array[first_drop]),
                        xytext=(first_drop + len(max_cc_array)*0.1, max_cc_array[first_drop]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Normalized MaxCC (percentage of original)
    normalized_cc = max_cc_array / max_cc_array[0] * 100
    ax2.plot(nodes_removed, normalized_cc, 'r-s', linewidth=2, markersize=4, alpha=0.7)
    ax2.set_xlabel('Number of Nodes Removed', fontweight='bold')
    ax2.set_ylabel('Largest CC Size (% of Original)', fontweight='bold')
    ax2.set_title(f'Normalized MaxCC Curve - Graph {graph_id}, Iter {iter_num}', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(normalized_cc))
    ax2.set_ylim(0, 105)
    
    # Add horizontal lines for key thresholds
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25% threshold')
    ax2.axhline(y=10, color='darkred', linestyle='--', alpha=0.7, label='10% threshold')
    ax2.legend()
    
    # Add statistics
    stats_text = f"""Statistics:
Original size: {max_cc_array[0]}
Final size: {max_cc_array[-1]}
Total nodes removed: {len(max_cc_array)}
Efficiency: {len(max_cc_array)/max_cc_array[0]:.3f} nodes/unit size"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
              verticalalignment='top', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved MaxCC curve plot: {plot_file_path}")
    
    plt.close()


def check_sol_in_graph(g, sol):
    missing_nodes = [node for node in sol if node not in g.nodes()]
    if missing_nodes:
        print(f"  ⚠ Warning: Some solution nodes ({missing_nodes}) are not in the graph")
        # Filter out missing nodes
        sol = [node for node in sol if node in g.nodes()]
        if not sol:
            print(f"  ⚠ No valid nodes in solution after filtering")
            return

def plot_max_cc_lists(all_MaxCCList, plot_file_path):
    """
    Plot all MaxCCList curves together in one figure (raw and normalized),
    following the style of plot_max_cc_curve.

    Parameters:
    - all_MaxCCList: list of lists. Each inner list is a MaxCC sequence for a graph
    - plot_file_path: output image path
    """
    print(f"  Creating comprehensive analysis...")

    if not all_MaxCCList:
        print("  ⚠ No MaxCC data provided, skipping")
        return

    # Filter out any empty sequences
    series_list = [np.asarray(seq, dtype=float) for seq in all_MaxCCList if seq is not None and len(seq) > 0]
    if not series_list:
        print("  ⚠ All MaxCC lists are empty, skipping")
        return

    # Build x ranges per series (they may have different lengths)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw MaxCC vs nodes removed
    max_first = 0.0
    max_len = 0
    for idx, arr in enumerate(series_list):
        x = np.arange(len(arr))
        ax1.plot(x, arr, linewidth=1.6, alpha=0.8, label=f'g{idx}')
        max_first = max(max_first, arr[0])
        max_len = max(max_len, len(arr))
    ax1.set_xlabel('Number of Nodes Removed', fontweight='bold')
    ax1.set_ylabel('Largest Connected Component Size', fontweight='bold')
    ax1.set_title('MaxCC Curves (all graphs)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if max_len > 0:
        ax1.set_xlim(0, max_len)
    if max_first > 0:
        ax1.set_ylim(0, max_first * 1.05)
    ax1.legend(fontsize=8, ncol=2)

    # Normalized (% of original)
    for idx, arr in enumerate(series_list):
        x = np.arange(len(arr))
        base = arr[0] if arr[0] != 0 else 1.0
        norm = arr / base * 100.0
        ax2.plot(x, norm, linewidth=1.6, alpha=0.8, label=f'g{idx}')
    ax2.set_xlabel('Number of Nodes Removed', fontweight='bold')
    ax2.set_ylabel('Largest CC Size (% of Original)', fontweight='bold')
    ax2.set_title('Normalized MaxCC Curves (all graphs)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if max_len > 0:
        ax2.set_xlim(0, max_len)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50%')
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25%')
    ax2.axhline(y=10, color='darkred', linestyle='--', alpha=0.7, label='10%')
    ax2.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive analysis: {plot_file_path}")
    plt.close()


def save_real_results(results, model_config, eval_config):
    """Save evaluation results to individual dataset CSV files
        results: dict of dicts, {dataset_name: {iter_num: {'score': score, 'time': time}}}
        model_config: model configuration
        eval_config: evaluation configuration

        Returns: None
    """
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='real')
    datasets = eval_config['datasets']

    # Create output directory if it doesn't exist
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    print(f"\nSaving evaluation results to: {save_result_dir}")

    # Save individual dataset files
    for dataset in datasets:
        dataset_file = os.path.join(save_result_dir, f"{dataset}.csv")
        
        dataset_data_for_df = []
        if dataset in results:
            for iter_num, data in results[dataset].items():
                dataset_data_for_df.append({'iter': iter_num, 'score': data['score'], 'time': data['time']})
        
        if dataset_data_for_df:
            df = pd.DataFrame(dataset_data_for_df)
            df = df.sort_values(by='iter').reset_index(drop=True)
            df.to_csv(dataset_file, index=False)
            print(f"  ✓ Saved {dataset}.csv with {len(df)} rows.")
        else:
            print(f"  ⚠ No valid data for dataset {dataset} to save.")
            
    print("✓ All dataset CSV files updated.")



