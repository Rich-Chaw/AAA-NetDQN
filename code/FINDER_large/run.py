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

def run_synthetic_evaluation(FINDER_type):
    """
    Run synthetic dataset evaluation with example parameters
    """
    
    # Example 1: Evaluate BA model trained on 30-50 nodes with m=3
    # on synthetic datasets with different nrange and m values
    print("=" * 80)
    

    # eval model from model path

    # for nrange in ['30_50',"50_100"]:
    #     for m in [1,2,3,4,5,6]:
    #         cmd = [
    #             "python", "./FINDER_ND/testSynthetic.py",
    #             "--model_path", f"./models/BA_nrange_{nrange}_m_{m}",
    #             "--min_iter", "3000",
    #             "--max_iter", "90000",
    #             "--iter_step", "3000",
    #             "--per_graphs", "2",
    #             "--eval_all_iters"
    #         ]
    #         subprocess.run(cmd)

    # cmd = [
    #         "python", "./FINDER_ND/testSynthetic.py",
    #         "--model_path", f"./models/BA_nrange_30_50_m_6",
    #         "--min_iter", "0",
    #         "--max_iter", "50000",
    #         "--iter_step", "600",
    #         "--per_graphs", "5",
    #         "--eval_all_iters"
    # ]
    # subprocess.run(cmd)

    # # eval iter 900,
    # cmd = [
    #     "python", "./FINDER_ND/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_2",
    #     "--eval_iter","900",
    #     "--per_graphs", "2"
    # ]
    # subprocess.run(cmd)

    # # eval_iter < 0, eval best iter,
    # cmd = [
    #     "python", "./FINDER_ND/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_2",
    #     "--eval_iter","-1",
    #     "--per_graphs", "2"
    # ]




    # !!! nrange and m still need to modify in or config.json by hand


    # Real

    # cmd = [
    #         "python", "./FINDER_ND/testReal.py",
    #         "--model_path", f"./models/BA_nrange_30_50_m_4",
    #         "--min_iter", "3000",
    #         "--max_iter", "3300",
    #         "--iter_step", "300",
    #         "--eval_all_iters"
    #     ]
    # subprocess.run(cmd)

    # cmd = [
    #         "python", "./FINDER_ND/testReal.py",
    #         "--eval_iter","108300"
    #     ]

    # find best iter as iter
    # cmd = [
    #         "python", "./FINDER_ND/testReal.py",
    #         "--eval_iter","-1"
    #     ]



    # cmd = [
    #         "python", "./FINDER_ND/testReal.py",
    #         "--eval_iter","108300",
    #         "--save_sol_only"
    #     ]
    
    # subprocess.run(cmd)


    cmd = [
            "python", f"./{FINDER_type}/testModels.py",
            "--eval_synth",
            "--eval_real",
            "--eval_all_iters"
        ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    FINDER_type = "FINDER_large"
    run_synthetic_evaluation(FINDER_type) 