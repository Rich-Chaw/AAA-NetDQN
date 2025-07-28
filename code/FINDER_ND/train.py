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

from GraphDQN import GraphDQN



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
    
    if not gpu_available:
        print("WARNING: No GPU detected. Training will be slow on CPU.")
    
    print("\n=== Starting GraphDQN Training ===")
    
    try:
        dqn = GraphDQN(
            g_type = 'barabasi_albert',
            gnn_model = 'GIN',
            target_graph = "Digg",
            num_min = 30,
            num_max = 50,
            ckpt_file = None
        )
        
        dqn.Train()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
