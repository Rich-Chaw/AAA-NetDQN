#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract actual router weights from trained MoE model
This script modifies the MoE model to expose router network outputs
"""

import sys, os

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

import networkx as nx
import pickle
from tqdm import tqdm
import argparse
import json


class RouterWeightExtractor:
    """Extract router weights from trained MoE model"""
    
    def __init__(self, dqn_model):
        self.dqn = dqn_model
        self.session = dqn_model.session
        self.build_router_extraction_ops()
    
    def build_router_extraction_ops(self):
        """Build TensorFlow operations to extract router weights"""
        print("Building router weight extraction operations...")
        
        # We need to access the router network from the encoder

        
        # Placeholders for graph features (same as router input)
        self.graph_features_ph = tf1.placeholder(tf.float32, [None, 4])
        
        # Rebuild router network (copy from MoEGraphDQN_modules.py)
        self.router_mlp1 = tf1.Variable(
            tf1.truncated_normal([4, 64], stddev=0.01), tf.float32
        )
        self.router_bias1 = tf1.Variable(tf.zeros([64]), tf.float32)
        
        self.router_mlp2 = tf1.Variable(
            tf1.truncated_normal([64, 4], stddev=0.01), tf.float32
        )
        self.router_bias2 = tf1.Variable(tf.zeros([4]), tf.float32)
        
        # Router forward pass
        h = tf.nn.relu(tf.matmul(self.graph_features_ph, self.router_mlp1) + self.router_bias1)
        expert_logits = tf.matmul(h, self.router_mlp2) + self.router_bias2
        
        # Get expert weights and top-k selection
        self.expert_weights = tf.nn.softmax(expert_logits, axis=-1)
        
        # Get top-k experts
        self.top_k_values, self.top_k_indices = tf.nn.top_k(self.expert_weights, k=2)
        
        # Create mask for top-k experts
        self.expert_mask = tf.reduce_sum(
            tf.one_hot(self.top_k_indices, depth=4, axis=-1), 
            axis=1
        )
        
        # Normalize weights for selected experts only
        masked_weights = self.expert_weights * self.expert_mask
        self.normalized_weights = masked_weights / (tf.reduce_sum(masked_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Extract trained router weights from the loaded model and assign to our variables
        self._assign_trained_weights()
        
        print("Router extraction operations built successfully")

    def _assign_trained_weights(self):
        """Extract trained router weights from the loaded model and assign to our variables"""
        print("Extracting trained router weights...")
        
        try:
            # Direct access to router variables through model object hierarchy
            # This is much more reliable than searching through all 72 trainable variables
            
            # Access the router through: dqn.encoder.router
            if hasattr(self.dqn, 'encoder') and hasattr(self.dqn.encoder, 'router'):
                router = self.dqn.encoder.router
                print("Found router object in dqn.encoder.router")
                
                # Get the actual TensorFlow variables from the router object
                router_mlp1_var = router.router_mlp1
                router_bias1_var = router.router_bias1
                router_mlp2_var = router.router_mlp2
                router_bias2_var = router.router_bias2
                
                print(f"Found router variables:")
                print(f"  router_mlp1: {router_mlp1_var.name} {router_mlp1_var.shape}")
                print(f"  router_bias1: {router_bias1_var.name} {router_bias1_var.shape}")
                print(f"  router_mlp2: {router_mlp2_var.name} {router_mlp2_var.shape}")
                print(f"  router_bias2: {router_bias2_var.name} {router_bias2_var.shape}")
                
                # Initialize our new variables first
                init_ops = tf1.variables_initializer([
                    self.router_mlp1, self.router_bias1, 
                    self.router_mlp2, self.router_bias2
                ])
                self.session.run(init_ops)
                
                # Create assignment operations to copy trained weights
                assign_ops = [
                    tf1.assign(self.router_mlp1, router_mlp1_var),
                    tf1.assign(self.router_bias1, router_bias1_var),
                    tf1.assign(self.router_mlp2, router_mlp2_var),
                    tf1.assign(self.router_bias2, router_bias2_var)
                ]
                
                # Execute assignment operations
                self.session.run(assign_ops)
                print("Successfully assigned all 4 router weight variables")
                
            else:
                print("Warning: Could not find router in dqn.encoder.router")
                print("Available attributes in dqn:", [attr for attr in dir(self.dqn) if not attr.startswith('_')])
                if hasattr(self.dqn, 'encoder'):
                    print("Available attributes in dqn.encoder:", [attr for attr in dir(self.dqn.encoder) if not attr.startswith('_')])
                
                # Fallback: try to find router variables by searching with better criteria
                self._fallback_weight_assignment()
                
        except Exception as e:
            print(f"Error during direct weight assignment: {e}")
            print("Trying fallback approach...")
            self._fallback_weight_assignment()
    
    def _fallback_weight_assignment(self):
        """Fallback method to find router weights when direct access fails"""
        print("Using fallback weight assignment...")
        
        try:
            # Get all trainable variables
            all_vars = tf1.trainable_variables()
            print(f"Total trainable variables: {len(all_vars)}")
            
            # More sophisticated search: look for variables with router-specific characteristics
            router_candidates = {
                'mlp1': [],
                'bias1': [],
                'mlp2': [],
                'bias2': []
            }
            
            for var in all_vars:
                shape = var.get_shape().as_list()
                name = var.name.lower()
                
                # Look for router-specific patterns in variable names
                if 'router' in name:
                    if shape == [4, 64]:
                        router_candidates['mlp1'].append(var)
                    elif shape == [64] and 'bias' in name:
                        router_candidates['bias1'].append(var)
                    elif shape == [64, 4]:
                        router_candidates['mlp2'].append(var)
                    elif shape == [4] and 'bias' in name:
                        router_candidates['bias2'].append(var)
                
                # If no 'router' in name, look for unique shape combinations
                elif shape == [4, 64] and len([v for v in all_vars if v.get_shape().as_list() == [4, 64]]) == 1:
                    # Only one variable with shape [4, 64] - likely router_mlp1
                    router_candidates['mlp1'].append(var)
                elif shape == [64, 4] and len([v for v in all_vars if v.get_shape().as_list() == [64, 4]]) == 1:
                    # Only one variable with shape [64, 4] - likely router_mlp2
                    router_candidates['mlp2'].append(var)
            
            # Print candidates found
            for key, candidates in router_candidates.items():
                print(f"Router {key} candidates: {[v.name for v in candidates]}")
            
            # Select the best candidates (prefer those with 'router' in name)
            selected_vars = {}
            for key, candidates in router_candidates.items():
                if candidates:
                    # Prefer variables with 'router' in name
                    router_named = [v for v in candidates if 'router' in v.name.lower()]
                    selected_vars[key] = router_named[0] if router_named else candidates[0]
                else:
                    selected_vars[key] = None
            
            # Initialize our variables
            init_ops = tf1.variables_initializer([
                self.router_mlp1, self.router_bias1, 
                self.router_mlp2, self.router_bias2
            ])
            self.session.run(init_ops)
            
            # Create assignment operations for found variables
            assign_ops = []
            if selected_vars['mlp1']:
                assign_ops.append(tf1.assign(self.router_mlp1, selected_vars['mlp1']))
                print(f"Assigning router_mlp1 from {selected_vars['mlp1'].name}")
            
            if selected_vars['bias1']:
                assign_ops.append(tf1.assign(self.router_bias1, selected_vars['bias1']))
                print(f"Assigning router_bias1 from {selected_vars['bias1'].name}")
            
            if selected_vars['mlp2']:
                assign_ops.append(tf1.assign(self.router_mlp2, selected_vars['mlp2']))
                print(f"Assigning router_mlp2 from {selected_vars['mlp2'].name}")
            
            if selected_vars['bias2']:
                assign_ops.append(tf1.assign(self.router_bias2, selected_vars['bias2']))
                print(f"Assigning router_bias2 from {selected_vars['bias2'].name}")
            
            if assign_ops:
                self.session.run(assign_ops)
                print(f"Successfully assigned {len(assign_ops)} router variables using fallback method")
            else:
                print("Warning: No suitable router variables found, using random initialization")
                
        except Exception as e:
            print(f"Error in fallback assignment: {e}")
            print("Using random initialization for all router variables")
            # Just initialize with random values
            init_ops = tf1.variables_initializer([
                self.router_mlp1, self.router_bias1, 
                self.router_mlp2, self.router_bias2
            ])
            self.session.run(init_ops)

    
    def extract_graph_features_batch(self, graphs):
        """Extract graph features for a batch of graphs"""
        features_batch = []
        
        for g in graphs:
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            
            if n_nodes <= 1:
                features = [0.0, 0.0, 0.0, 0.0]
            else:
                # Calculate features similar to MoE encoder
                density = nx.density(g)
                avg_degree = 2 * n_edges / n_nodes
                
                # Normalize features (same as in MoEGraphEncoder)
                normalized_nodes = n_nodes / 200.0
                estimated_m = avg_degree / 2.0
                normalized_m = np.clip(estimated_m / 10.0, 0.0, 1.0)
                normalized_density = np.clip(density, 0.0, 1.0)
                normalized_avg_degree = np.clip(avg_degree / 20.0, 0.0, 1.0)
                
                features = [
                    normalized_nodes,
                    normalized_m,
                    normalized_density,
                    normalized_avg_degree
                ]
            
            features_batch.append(features)
        
        return np.array(features_batch)
    
    def get_router_weights(self, graphs):
        """Get actual router weights from the trained model for a list of graphs"""
        # Extract graph features
        graph_features = self.extract_graph_features_batch(graphs)
        
        # Use the trained router network to get actual routing decisions
        feed_dict = {self.graph_features_ph: graph_features}
        
        results = self.session.run([
            self.expert_weights,
            self.normalized_weights,
            self.top_k_indices,
            self.expert_mask
        ], feed_dict=feed_dict)
        
        expert_weights, normalized_weights, top_k_indices, expert_mask = results
        
        return {
            'graph_features': graph_features,
            'expert_weights': expert_weights,
            'normalized_weights': normalized_weights,
            'top_k_indices': top_k_indices,
            'expert_mask': expert_mask
        }

    
    def analyze_routing_patterns(self, graphs, labels, graph_type="unknown"):
        """Analyze routing patterns for a set of graphs"""
        print(f"Analyzing routing patterns for {len(graphs)} {graph_type} graphs...")
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_results = []
        
        for i in tqdm(range(0, len(graphs), batch_size), desc="Processing batches"):
            batch_graphs = graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            try:
                # Get router weights for this batch
                router_results = self.get_router_weights(batch_graphs)
                
                # Store results with metadata
                for j, (g, label) in enumerate(zip(batch_graphs, batch_labels)):
                    result = {
                        'graph_type': graph_type,
                        'graph_label': label,
                        'n_nodes': g.number_of_nodes(),
                        'n_edges': g.number_of_edges(),
                        'density': nx.density(g) if g.number_of_nodes() > 1 else 0.0,
                        'avg_degree': 2 * g.number_of_edges() / g.number_of_nodes() if g.number_of_nodes() > 0 else 0.0,
                        'graph_features': router_results['graph_features'][j],
                        'expert_weights': router_results['expert_weights'][j],
                        'normalized_weights': router_results['normalized_weights'][j],
                        'top_k_experts': router_results['top_k_indices'][j],
                        'selected_expert': np.argmax(router_results['expert_weights'][j]),
                        'max_weight': np.max(router_results['expert_weights'][j]),
                        'entropy': -np.sum(router_results['expert_weights'][j] * 
                                         np.log(router_results['expert_weights'][j] + 1e-8))
                    }
                    all_results.append(result)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results, save_path):
        """Save routing analysis results"""
        print(f"Saving results to {save_path}")
        
        # Save as pickle for full data
        with open(save_path.replace('.pkl', '_full.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save as CSV for easy analysis
        import pandas as pd
        
        # Flatten results for CSV
        csv_data = []
        for result in results:
            csv_row = {
                'graph_type': result['graph_type'],
                'graph_label': result['graph_label'],
                'n_nodes': result['n_nodes'],
                'n_edges': result['n_edges'],
                'density': result['density'],
                'avg_degree': result['avg_degree'],
                'selected_expert': result['selected_expert'],
                'max_weight': result['max_weight'],
                'entropy': result['entropy'],
            }
            
            # Add individual expert weights
            for i in range(4):
                csv_row[f'expert_{i}_weight'] = result['expert_weights'][i]
                csv_row[f'expert_{i}_normalized'] = result['normalized_weights'][i]
            
            # Add graph features
            feature_names = ['norm_nodes', 'norm_m', 'norm_density', 'norm_avg_degree']
            for i, name in enumerate(feature_names):
                csv_row[name] = result['graph_features'][i]
            
            # Add top-k experts
            csv_row['top_k_expert_1'] = result['top_k_experts'][0]
            csv_row['top_k_expert_2'] = result['top_k_experts'][1]
            
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(save_path.replace('.pkl', '.csv'), index=False)
        
        print(f"Results saved to {save_path.replace('.pkl', '_full.pkl')} and {save_path.replace('.pkl', '.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Extract router weights from MoE model')
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--save_dir", type=str, default="./router_weights",
                       help="Directory to save extracted weights")
    parser.add_argument("--n_synthetic", type=int, default=50,
                       help="Number of synthetic graphs per configuration")
    parser.add_argument("--real_datasets", type=str, nargs='+', 
                       default=['Crime','Digg'],
                       help="Real datasets to analyze")
    
    args = parser.parse_args()

    # -----------------------------------------LOAD modules----------------------------------------------
    FINDER_type = args.model_path.split("/")[1]
    # sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    if "moe" in FINDER_type:
        sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        sys.path.append(os.path.dirname(__file__) + os.sep + f'FINDER/')
        from GraphDQN import GraphDQN
        from MoEGraphDQN import MoEGraphDQN
    else:
        sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        from GraphDQN import GraphDQN
    from testUtils import load_config, create_moe_model, load_synthetic_graphs, load_real_graph


    # Load configuration and create model
    config = load_config()
    model_config = config['model_config']
    data_config = config['data_config']
    
    # Update model config based on model_path
    model_config = config['model_config']
    model_path_config_file = os.path.join(args.model_path,"config.json")
    if os.path.exists(model_path_config_file):
        # Parse model path to extract parameters
        with open(model_path_config_file, 'r') as f:
            model_config.update(json.load(f)["model_config"])
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Router Weight Extraction")
    print("=" * 60)
    
    try:
        print("\nLoading MoE model...")
        dqn = create_moe_model(model_config)
        best_ckpt_file = dqn.findModel()
        dqn.LoadModel(best_ckpt_file)
        print(f"Loaded model: {best_ckpt_file}")
        
        # Initialize extractor
        extractor = RouterWeightExtractor(dqn)
        
        all_results = []
        
        # Extract weights for synthetic graphs
        print("\nProcessing synthetic graphs...")
        synthetic_configs = [
            {'nrange': '30_50', 'm': 2},
            {'nrange': '30_50', 'm': 4},
            {'nrange': '50_100', 'm': 2},
            {'nrange': '50_100', 'm': 4},
            {'nrange': '50_100', 'm': 6},
        ]
        
        for config in synthetic_configs:
            print(f"  Processing BA graphs with {config}...")
            try:
                graphs = load_synthetic_graphs('BA', g_num=args.n_synthetic, **config)
                labels = [f"BA_nrange_{config['nrange']}_m_{config['m']}_{i}" 
                         for i in range(len(graphs))]
                
                results = extractor.analyze_routing_patterns(
                    graphs, labels, f"synthetic_BA_{config['nrange']}_m_{config['m']}"
                )
                all_results.extend(results)
                print(f"    Processed {len(results)} graphs")
                
            except Exception as e:
                print(f"    Error processing {config}: {e}")
        
        # Extract weights for real graphs
        print("\nProcessing real graphs...")
        dataset_dir = os.path.join(data_config['dataset_dir'], "real")
        
        for dataset in args.real_datasets:
            print(f"  Processing {dataset}...")
            try:
                g = load_real_graph(dataset, dataset_dir)
                if g is not None:
                    results = extractor.analyze_routing_patterns([g], [dataset], f"real_{dataset}")
                    all_results.extend(results)
                    print(f"    Processed {dataset}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
                else:
                    print(f"    Failed to load {dataset}")
            except Exception as e:
                print(f"    Error processing {dataset}: {e}")
        
        # Save all results
        save_path = os.path.join(args.save_dir, "router_weights_analysis.pkl")
        extractor.save_results(all_results, save_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Router Weight Extraction Summary")
        print(f"{'='*60}")
        print(f"Total graphs processed: {len(all_results)}")
        
        if all_results:
            # Expert usage summary
            expert_counts = {}
            for result in all_results:
                expert = result['selected_expert']
                expert_counts[expert] = expert_counts.get(expert, 0) + 1
            
            print("\nExpert Usage:")
            for expert in range(4):
                count = expert_counts.get(expert, 0)
                percentage = count / len(all_results) * 100
                print(f"  Expert {expert}: {count} graphs ({percentage:.1f}%)")
            
            # Confidence summary
            max_weights = [result['max_weight'] for result in all_results]
            entropies = [result['entropy'] for result in all_results]
            
            print(f"\nRouter Confidence:")
            print(f"  Average max weight: {np.mean(max_weights):.3f} ± {np.std(max_weights):.3f}")
            print(f"  Average entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        
        print(f"\nResults saved to: {args.save_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()