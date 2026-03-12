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

import subprocess

from MoEGraphDQN import MoEGraphDQN
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

def main():
    # Check GPU setup first
    gpu_available = check_gpu_setup()
    
    # setup cython modules
    # subprocess.run('python -u .\\FINDER_moe\\setup.py build_ext -i')
    import argparse
    parser = argparse.ArgumentParser(description='train MoE models')
    parser.add_argument("--from_pretrained", action="store_true", help="plot all iter results in csv from result dir")
    args = parser.parse_args()


    print("\n=== Starting MoE GraphDQN Training ===")

    # Test basic MoE creation
    model_config = {
        'g_type': 'BA',
        'g_params': {'nrange': '50_100', 'm': 6},
        'gnn_model': 'graphSage',
        'save_model_dir': './FINDER_moe_v2/models',
        'moe_config': {
            'num_experts': 4,
            'top_k': 2,
            'router_dropout': 0.1,
            'load_balance_loss_weight': 0.01,
            'from_pretrained': args.from_pretrained,  # Start with False for testing
            'pretrained_model_dir': './FINDER/models'
        }
    }

    
    try:
        moe_dqn = MoEGraphDQN(
            **model_config
        )
        
        moe_config = model_config['moe_config']
        print("MoE Configuration:")
        print(f"  - Number of experts: {moe_dqn.num_experts}")
        print(f"  - Top-K experts: {moe_config['top_k']}")
        print(f"  - Load balance weight: {moe_config['load_balance_loss_weight']}")
        print(f"  - Router dropout: {moe_config['router_dropout']}")
        print(f"  - from_pretrained: {moe_config['from_pretrained']}")
        if moe_config['from_pretrained']:
            print(f"  - pretrained_model_dir: {moe_config['pretrained_model_dir']}")

        moe_dqn.Train()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
