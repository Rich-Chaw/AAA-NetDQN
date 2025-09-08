#!/usr/bin/env python3
"""
Configuration file for FINDER_ND fine-tuning
Contains different configurations for various fine-tuning scenarios
"""

# Base configuration for fine-tuning
BASE_CONFIG = {
    'pretrained_ckpt_path': './models/BA_sota/graphSage_nrange_30_50_iter_78000.ckpt',
    'embedding_method': 'graphSage',
    'embedding_size': 64,
    'feature_size': 2,
    'reg_hidden': 64,
    'aux_dim': 4,
    'initialization_stddev': 0.01,
    'gnn_layers': 3,
    'learning_rate': 0.001,
    'batch_size': 16,
    'num_epochs': 50
}

# Configuration for Digg dataset
DIGG_CONFIG = {
    **BASE_CONFIG,
    'real_graph_path': '../../dataset/real/Digg.txt',
    'learning_rate': 0.0005,  # Lower learning rate for large graph
    'batch_size': 8,  # Smaller batch size due to memory constraints
    'num_epochs': 100
}

# Configuration for Crime dataset (smaller graph)
CRIME_CONFIG = {
    **BASE_CONFIG,
    'real_graph_path': '../../dataset/real/Crime.txt',
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50
}

# Configuration for Enron dataset
ENRON_CONFIG = {
    **BASE_CONFIG,
    'real_graph_path': '../../dataset/real/Enron.txt',
    'learning_rate': 0.0005,
    'batch_size': 16,
    'num_epochs': 75
}

# Configuration for Facebook dataset (large graph)
FACEBOOK_CONFIG = {
    **BASE_CONFIG,
    'real_graph_path': '../../dataset/real/Facebook.txt',
    'learning_rate': 0.0001,  # Very low learning rate for large graph
    'batch_size': 4,  # Very small batch size
    'num_epochs': 150
}

# Freezing mode specific configurations
FREEZE_CONFIGS = {
    'Digg_freeze': {**DIGG_CONFIG, 'fine_tune_mode': 'freeze'},
    'Crime_freeze': {**CRIME_CONFIG, 'fine_tune_mode': 'freeze'},
    'Enron_freeze': {**ENRON_CONFIG, 'fine_tune_mode': 'freeze'},
    'Facebook_freeze': {**FACEBOOK_CONFIG, 'fine_tune_mode': 'freeze'}
}

# Full fine-tuning mode specific configurations
FULL_CONFIGS = {
    'Digg_full': {**DIGG_CONFIG, 'fine_tune_mode': 'full'},
    'Crime_full': {**CRIME_CONFIG, 'fine_tune_mode': 'full'},
    'Enron_full': {**ENRON_CONFIG, 'fine_tune_mode': 'full'},
    'Facebook_full': {**FACEBOOK_CONFIG, 'fine_tune_mode': 'full'}
}

# All configurations
ALL_CONFIGS = {**FREEZE_CONFIGS, **FULL_CONFIGS}

def get_config(config_name):
    """Get configuration by name"""
    if config_name in ALL_CONFIGS:
        return ALL_CONFIGS[config_name]
    else:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(ALL_CONFIGS.keys())}")

def list_configs():
    """List all available configurations"""
    print("Available fine-tuning configurations:")
    print("\nFreezing mode configurations:")
    for name in FREEZE_CONFIGS.keys():
        print(f"  - {name}")
    
    print("\nFull fine-tuning mode configurations:")
    for name in FULL_CONFIGS.keys():
        print(f"  - {name}")
    
    print("\nUsage:")
    print("  from finetune_config import get_config")
    print("  config = get_config('digg_freeze')")

if __name__ == "__main__":
    list_configs() 