#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

# Suppress warnings and verbose output
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use first GPU

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings

# Suppress other verbose libraries
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from AdvanceGraphDQN import AdvanceGraphDQN
print(sys.path)


def check_gpu_setup():
    """Check GPU setup before training"""
    print("=== GPU Setup Check ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU device(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        return True
    else:
        print("No GPU devices found!")
        print("Training will use CPU only.")
        return False

# def save_config(config):
#     import json
#     import os
    
#     model_config = config['model_config']
#     save_model_dir = model_config['save_model_dir']
#     gnn_model = model_config['gnn_model']
#     g_type = model_config['g_type']
#     nrange = model_config['g_params']['nrange']
#     m = model_config['g_params']['m']
    
#     # Create the model-specific directory name
#     model_dir_name = f"{gnn_model}_{g_type}_nrange_{nrange}_m_{m}"
#     save_config_path = os.path.join(save_model_dir, model_dir_name)
    
#     # Create directory if it doesn't exist
#     os.makedirs(save_config_path, exist_ok=True)
    
#     # Save config as JSON file
#     config_file_path = os.path.join(save_config_path, "train_config.json")
#     with open(config_file_path, 'w') as f:
#         json.dump(config, f, indent=2)
    
#     print(f"Config saved to: {config_file_path}")
#     return save_config_path

def main():
    # Check GPU setup first
    gpu_available = check_gpu_setup()
    config = {}
    #-------------------------------train config---------------------------------------------------
    
    # Example 1: Standard BA graph training
    # model_config = {
    #     'g_type': 'BA',
    #     'g_params': {
    #         'nrange': '50_70',
    #         'm': 6
    #     },
    #     'gnn_model': 'graphSage',
    #     'save_model_dir': './FINDER_advance/models',
    #     'options':{
    #         'IsDoubleDQN': False,
    #         'IsFeatures': False,
    #         'IsPrioritizedSampling': False,
    #         'IsDisturbG': False  # No augmentation
    #     }
    # }
    
    # # Example 2: Mixed graph types training
    model_config = {
        'g_type': 'mix',
        'g_params': {
            'nrange': '50_70',
            'mix_weights': [0.4, 0.1, 0.3, 0.2]  # [BA, ER, PL, SW] probabilities
        },
        'gnn_model': 'MoE',
        'save_model_dir': './FINDER_advance/models',
        'options':{
            'IsDoubleDQN': False,
            'IsFeatures': False,
            'IsPrioritizedSampling': False,
            'IsDisturbG': False
        },
        'save_config':True
    }
    
    # Example 3: BA graphs with augmentation
    # model_config = {
    #     'g_type': 'BA',
    #     'g_params': {
    #         'nrange': '50_70',
    #         'm': 6,
    #         'augmentation': {
    #             'drop_edge_prob': 0.05,      # 5% edges randomly dropped
    #             'add_edge_prob': 0.03,       # 3% new edges randomly added
    #             'use_random_walk': False,    # Don't use random walk (too aggressive)
    #             'ensure_connected': True,    # Keep graph connected
    #             'aug_probability': 0.5       # Only augment 50% of training graphs
    #         }
    #     },
    #     'gnn_model': 'graphSage',
    #     'save_model_dir': './FINDER_advance/models',
    #     'options':{
    #         'IsDoubleDQN': False,
    #         'IsFeatures': True,
    #         'IsPrioritizedSampling': False,
    #         'IsDisturbG': True  # Enable augmentation
    #     }
    # }
    
    # Example 4: Mixed graphs with augmentation (most robust training)
    # model_config = {
    #     'g_type': 'mix',
    #     'g_params': {
    #         'nrange': '50_70',
    #         'm': 6,
    #         'mix_weights': [0.3, 0.2, 0.2, 0.3],  # [BA, ER, PL, SW]
    #         'augmentation': {
    #             'drop_edge_prob': 0.05,
    #             'add_edge_prob': 0.03,
    #             'use_random_walk': False,
    #             'ensure_connected': True,
    #             'aug_probability': 0.5
    #         }
    #     },
    #     'gnn_model': 'graphSage',
    #     'save_model_dir': './FINDER_advance/models',
    #     'options':{
    #         'IsDoubleDQN': False,
    #         'IsFeatures': True,
    #         'IsPrioritizedSampling': False,
    #         'IsDisturbG': True
    #     }
    # }

    # model_config = {
    #     'g_type': 'ego',
    #     'g_params': {'nrange': '100_200',
    #                  'target_graph': 'Digg',
    #                  'train_dir': f"../../dataset/synthetic/GSDM",
    #                  'valid_dir': f"../../dataset/synthetic/GSDM",
    #                  "dataset_id":0                                 # train ego graph id,begin with 0
    #                  },
    #     'gnn_model': 'graphSage',
    #     'save_model_dir': './FINDER/models'
    # }
    
    #----------------------------train-----------------------------------------
    print("\n=== Starting AdvanceGraphDQN Training ===")
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}")
    
    
    try:
        dqn = AdvanceGraphDQN(
            **model_config
        )
        dqn.Train()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
