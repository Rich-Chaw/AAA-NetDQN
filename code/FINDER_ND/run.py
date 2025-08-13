#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to run synthetic dataset evaluation using testSynthetic.py

This script demonstrates how to evaluate a trained model on synthetic datasets
with different parameters (nrange and m values).
"""

import subprocess
import sys
import os

def run_synthetic_evaluation():
    """
    Run synthetic dataset evaluation with example parameters
    """
    
    # Example 1: Evaluate BA model trained on 30-50 nodes with m=3
    # on synthetic datasets with different nrange and m values
    print("=" * 80)
    

    # eval model from model path

    # cmd1 = [
    #     "python", "./FINDER_ND/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_3",
    #     "--min_iter", "0",
    #     "--max_iter", "300",
    #     "--iter_step", "300",
    #     "--per_graphs", "5"
    # ]
    
    # subprocess.run(cmd1)

    # cmd2 = [
    #     "python", "./FINDER_ND/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_2",
    #     "--min_iter", "0",
    #     "--max_iter", "300",
    #     "--iter_step", "300",
    #     "--per_graphs", "10"
    # ]
    # subprocess.run(cmd2)

    # eval iter 900,
    cmd3 = [
        "python", "./FINDER_ND/testSynthetic.py",
        "--model_path", "./models/BA_nrange_30_50_m_2",
        "--eval_iter","900",
        "--per_graphs", "2"
    ]
    subprocess.run(cmd3)

    # !!! nrange and m still need to modify by hand

if __name__ == "__main__":
    run_synthetic_evaluation() 