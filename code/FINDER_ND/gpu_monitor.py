#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time

def check_gpu_availability():
    """Check if GPU is available and print device information"""
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    if tf.config.list_physical_devices('GPU'):
        print("GPU devices found:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
        
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"GPU computation test: {c}")
            print("GPU computation successful!")
    else:
        print("No GPU devices found!")

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        # This requires tensorflow-gpu and proper CUDA installation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                print(f"GPU: {gpu}")
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Error monitoring GPU memory: {e}")

if __name__ == "__main__":
    check_gpu_availability()
    monitor_gpu_memory() 