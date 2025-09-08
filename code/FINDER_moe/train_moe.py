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
    
    print("\n=== Starting MoE GraphDQN Training ===")
    
    # MoE configuration
    moe_config = {
        'num_experts': 4,
        'top_k': 2,
        'router_dropout': 0.1,
        'load_balance_loss_weight': 0.01
    }
    
    try:
        moe_dqn = MoEGraphDQN(
            g_type = 'BA',
            g_params = {'nrange': '50_100',
                        'm':6},
            target_graph = "Digg",
            moe_config = moe_config
        )
        
        print("MoE Configuration:")
        print(f"  - Number of experts: {moe_config['num_experts']}")
        print(f"  - Top-K experts: {moe_config['top_k']}")
        print(f"  - Load balance weight: {moe_config['load_balance_loss_weight']}")
        print(f"  - Router dropout: {moe_config['router_dropout']}")
        
        moe_dqn.Train()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
