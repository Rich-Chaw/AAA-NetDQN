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
from testUtils import load_config, create_model, load_synthetic_graphs,detailed_result_dir
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

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

def create_comprehensive_analysis(all_MaxCCList, plot_file_path):
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

def evaluate_checkpoint_on_synthetic_datasets(dqn, checkpoint_iter, eval_config):
    """
    Evaluate a single checkpoint on all synthetic dataset configurations
    
    Parameters:
    - dqn: GraphDQN model instance
    - checkpoint_iter: iteration number to evaluate

    Returns:
    - dict: results for each dataset configuration
    """

    print(f"  Evaluating checkpoint {checkpoint_iter}...")

    # Load the specific checkpoint
    dqn.LoadModel(f"{dqn.embeddingMethod}_iter_{checkpoint_iter}.ckpt")
    
    results = {}
    
    # Evaluate on each synthetic dataset configuration
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            print(f"    Testing {config_key}...")
            
            try:
                # Load synthetic graphs for this configuration
                graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                        g_num = eval_config["per_graphs"],
                                        nrange = nrange, 
                                        m=m)
                
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
    Save synthetic evaluation results to CSV files
    
    Parameters:
    - all_results: dict with iteration -> results mapping
    - eval_config: evaluation configuration
    - model_config: model configuration
    - save_dir: base directory to save results
    """
    # Create save directory structure
    # gnn_step = f"{model_config['gnn_model']}_StepRatio_{eval_config['step_ratio']:.4f}" #graphSage_StepRatio_0.0100
    # model_name = f"{model_config['g_type']}_nrange_{model_config['g_params']['num_min']}_{model_config['g_params']['num_max']}_m_{model_config['g_params']['m']}"
    # save_dir = os.path.join(save_dir, 'synthetic', model_name,gnn_step)
    save_result_dir = detailed_result_dir(model_config,eval_config,mode='synthetic')
    
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)
    
    print(f"\nSaving results to: {save_result_dir}")
    
    # Save results for each dataset configuration
    for nrange in eval_config['synthetic_nranges']:
        for m in eval_config['synthetic_m_values']:
            config_key = f"nrange_{nrange}_m_{m}"
            eval_g_type = eval_config['synthetic_g_type']
            eval_per_graphs = eval_config['per_graphs']
            csv_filename = f"{eval_g_type}_nrange_{nrange}_m_{m}_gn_{eval_per_graphs}.csv"
            csv_path = os.path.join(save_result_dir, csv_filename)
            
            # Prepare data for CSV
            data = {
                'iter': [],
                'score': [],
                'time': []
            }
            
            for iter_num in sorted(all_results.keys()):
                if all_results[iter_num] and config_key in all_results[iter_num]:
                    result = all_results[iter_num][config_key]
                    data['iter'].append(iter_num)
                    data['score'].append(result['score'])
                    data['time'].append(result['time'])
            
            # Create DataFrame and save
            if data['iter']:
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
                print(f"  ✓ Saved {csv_filename}: {len(df)} rows")
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
    parser.add_argument("--eval_iter",type=int,)
    args = parser.parse_args()
    
    # Update config with command line arguments
    eval_config = config['eval_config']
    eval_config.update({
        'min_iter': args.min_iter,
        'max_iter': args.max_iter,
        'iter_step': args.iter_step,
        'per_graphs': args.per_graphs,
        'synthetic_nranges': ["30_50", "50_100", "100_200", "200_300", "300_400", "400_500"],
        'synthetic_m_values': [1,2,3,4,5,6]
    })
    
    

    # Update model config if model_path is provided
    model_config = config['model_config']
    if args.model_path:
        # Parse model path to extract parameters
        path_parts = args.model_path.split('/')[-1].split('_')
        if len(path_parts) >= 6:
            model_config['g_type'] = path_parts[0]
            model_config['g_params']['num_min'] = int(path_parts[2])
            model_config['g_params']['num_max'] = int(path_parts[3])
            model_config['g_params']['m'] = int(path_parts[5])
            print(f"✓ Updated model config: {model_config}")
    
    print(f"\nSynthetic Dataset Evaluation")
    print(f"Target model: {model_config['g_type']}_nrange_{model_config['g_params']['num_min']}_{model_config['g_params']['num_max']}_m_{model_config['g_params']['m']}")
    
    print(f"\nStarting evaluation...")
    if args.eval_iter == None:
        # --------------------------eval all iters ----------------------------------
        # Generate target iterations
        target_iters = list(range(eval_config['min_iter'], eval_config['max_iter'] + 1, eval_config['iter_step']))
        # Create model instance
        dqn = create_model(model_config)
        
        # Evaluate all checkpoints
        print(f"Iterations: {eval_config['min_iter']} to {eval_config['max_iter']} (step: {eval_config['iter_step']})")
        print(f"Dataset configurations: {len(eval_config['synthetic_nranges'])} nrange × {len(eval_config['synthetic_m_values'])} m = {len(eval_config['synthetic_nranges']) * len(eval_config['synthetic_m_values'])} total")
        print(f"Graphs per: {eval_config['per_graphs']}")
        all_results = {}
        for iter_num in tqdm(target_iters, desc="Evaluating checkpoints"):
            try:
                results = evaluate_checkpoint_on_synthetic_datasets(dqn, iter_num, eval_config)
                if results:
                    all_results[iter_num] = results
                else:
                    all_results[iter_num] = None
            except Exception as e:
                print(f"✗ Error evaluating iteration {iter_num}: {e}")
                all_results[iter_num] = None
        # Save results
        save_synthetic_results(all_results, eval_config, model_config)
    
    else:
        # --------------------------eval specified iter, draw sol and CC curve ----------------------------------
        iter = args.eval_iter
        dqn = create_model(model_config,iter)

        # eval_g_type = 'BA'
        # nrange_list = ["30_50"]  # Single nrange
        # m_list = [1, 2, 3, 4, 5]
        # per_graphs = 1
        for nrange in eval_config["synthetic_nranges"]:
            for m in eval_config["synthetic_m_values"]:
                graphs = load_synthetic_graphs(eval_config["synthetic_g_type"],
                                        g_num = eval_config["per_graphs"],
                                        nrange = nrange, 
                                        m=m)
                all_MaxCCList = []
                save_result_dir = detailed_result_dir(model_config,eval_config,mode='synthetic')
                for i,g in enumerate(graphs):
                    general_result_file = os.path.join(save_result_dir,f"{eval_config['synthetic_g_type']}_nrange_{nrange}_m_{m}_g_{i}_iter_{iter}")
                    
                    # Get sol and MaxCCList
                    temp_sol_file = f"{general_result_file}.txt"
                    # sol = effective solution = nodes to remove
                    sol, sol_time = dqn.EvaluateRealData(g, temp_sol_file, eval_config['step_ratio'])
                    # print(sol)
                    # Evaluate sol , solution = sol_reinsert + sol_left
                    score, MaxCCList = dqn.EvaluateSol(g, temp_sol_file, eval_config['strategy_id'], reInsertStep=0.001)
                    all_MaxCCList.append(MaxCCList)

                    # Visualize dismantling process
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
                create_comprehensive_analysis(all_MaxCCList,os.path.join(save_result_dir,f"{eval_config['synthetic_g_type']}_nrange_{nrange}_m_{m}_iter_{iter}_MaxCC_comparison.png"))
    
    print(f"\n✓ Evaluation completed!")




if __name__ == "__main__":
    main()
