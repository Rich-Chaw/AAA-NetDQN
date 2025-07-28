#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script showing how to use the configuration file to create and use a GraphDQN model.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_graphdqn_from_config(config):
    """Create GraphDQN model from configuration"""
    try:
        from GraphDQN import GraphDQN
        model_config = config['model_config']
        dqn = GraphDQN(
            g_type=model_config['g_type'],
            gnn_model=model_config['gnn_model'],
            target_graph=model_config['target_graph'],
            num_min=model_config['num_min'],
            num_max=model_config['num_max'],
            ckpt_file=model_config['ckpt_file']
        )
        return dqn
    except ImportError as e:
        print(f"Error importing GraphDQN: {e}")
        return None
    except Exception as e:
        print(f"Error creating GraphDQN model: {e}")
        return None

def main():
    """Main function demonstrating configuration usage"""
    
    # Load configuration
    print("Loading configuration from config.json...")
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
    except FileNotFoundError:
        print("✗ Error: config.json not found")
        return
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in config.json: {e}")
        return
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print("=" * 50)
    model_config = config['model_config']
    print(f"Model Type: {model_config['g_type']}")
    print(f"GNN Model: {model_config['gnn_model']}")
    print(f"Target Graph: {model_config['target_graph']}")
    print(f"Node Range: {model_config['num_min']} - {model_config['num_max']}")
    print(f"Checkpoint: {model_config['ckpt_file']}")
    
    eval_config = config['evaluation_config']
    print(f"Datasets: {eval_config['datasets']}")
    print(f"Step Ratio: {eval_config['step_ratio']}")
    print(f"Strategy ID: {eval_config['strategy_id']}")
    
    # Create GraphDQN model
    print("\nCreating GraphDQN model...")
    dqn = create_graphdqn_from_config(config)
    
    if dqn is None:
        print("✗ Failed to create GraphDQN model")
        return
    
    print("✓ GraphDQN model created successfully")
    
    # Example of using the model
    print("\nModel Information:")
    print(f"  - Graph Type: {dqn.g_type}")
    print(f"  - GNN Model: {dqn.embeddingMethod}")
    print(f"  - Target Graph: {dqn.target_graph}")
    print(f"  - Node Range: {dqn.num_min} - {dqn.num_max}")
    print(f"  - Checkpoint File: {dqn.ckpt_file}")
    
    print("\n✓ Configuration system working correctly!")

if __name__ == "__main__":
    main() 