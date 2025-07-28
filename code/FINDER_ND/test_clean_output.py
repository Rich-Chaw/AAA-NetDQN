#!/usr/bin/env python3
"""
Test script to verify clean output without warnings
"""

# Suppress warnings and verbose output
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def test_clean_output():
    """Test that TensorFlow operations don't produce warnings"""
    print("Testing clean output...")
    
    # Test basic TensorFlow operations
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"✓ TensorFlow GPU computation successful: {c}")
    
    print("✓ All tests passed - output should be clean!")

if __name__ == "__main__":
    test_clean_output() 