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
import argparse

def run_synthetic_evaluation(FINDER_TYPE):
    """
    Run synthetic dataset evaluation with example parameters
    """
    
    # Example 1: Evaluate BA model trained on 30-50 nodes with m=3
    # on synthetic datasets with different nrange and m values

    
    # eval model from model path

    # for nrange in ['30_50',"50_100"]:
    #     for m in [1,2,3,4,5,6]:
    #         cmd = [
    #             "python", f"./{FINDER_TYPE}/testSynthetic.py",
    #             "--model_path", f"./models/BA_nrange_{nrange}_m_{m}",
    #             "--min_iter", "3000",
    #             "--max_iter", "90000",
    #             "--iter_step", "3000",
    #             "--per_graphs", "2",
    #             "--eval_all_iters"
    #         ]
    #         subprocess.run(cmd)

    # cmd = [
    #         "python", f"./{FINDER_TYPE}/testSynthetic.py",
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
    #     "python", f"./{FINDER_TYPE}/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_2",
    #     "--eval_iter","900",
    #     "--per_graphs", "2"
    # ]
    # subprocess.run(cmd)

    # # eval_iter < 0, eval best iter,
    # cmd = [
    #     "python", f"./{FINDER_TYPE}/testSynthetic.py",
    #     "--model_path", "./models/BA_nrange_30_50_m_2",
    #     "--eval_iter","-1",
    #     "--per_graphs", "2"
    # ]




    # !!! nrange and m still need to modify in or config.json by hand


    


    

def run_real_evaluation(FINDER_TYPE):
    # Real
    cmd = []
    model_path = f"./{FINDER_TYPE}/models/MoE_BA_nrange_50_100_m_6"
    
    # # eval best iter,default
    # cmd = [
    #         "python", f"./testReal.py",
    #         "--model_path", model_path,
    #     ]
    # subprocess.run(cmd)


    # eval specified iter
    # cmd = [
    #         "python", f"./testReal.py",
    #         "--model_path",model_path,
    #         "--eval_iter","3000"
    #     ]
    # subprocess.run(cmd)

    # eval specified iter,save sol and max_cc_list only, do not record in csv
    # cmd = [
    #         "python", f"./testReal.py",
    #         "--model_path", model_path,
    #         "--eval_iter","108300",
    #         "--save_sol_only"
    #     ]
    # subprocess.run(cmd)

    # eval iters
    cmd = [
            "python", f"./testReal.py",
            "--model_path", model_path,
            "--min_iter", "9900",
            "--max_iter", "40000",
            "--iter_step", "900",
            "--eval_all_iters"
        ]
    subprocess.run(cmd)


def run_models_evaluation(FINDER_TYPE):
    cmd = []
    cmd = [
            "python", f"./testModels.py",
            "--FINDER_type",f"{FINDER_TYPE}",
            # "--eval_synthetic",
            "--eval_real",
            # "--eval_all_iters"
        ]
    subprocess.run(cmd)

def run_router_evaluation(FINDER_TYPE):
    cmd = [
            "python", f"./testRouter.py",
            "--model_path",f"./{FINDER_TYPE}/models/MoE_BA_nrange_50_100_m_6",
            "--analysis_dir","./router_analysis_results",
            "--n_synthetic","1",
            "--real_datasets","Crime","HI-II-I4","Digg","Enron","Gnutella3","Epinions","Facebook"
        ]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete evaluation Pipeline')
    # parser.add_argument("--model_path", type=str, required=True,
    #                    help="Path to trained MoE model directory")
    parser.add_argument("--testReal", action="store_true")
    parser.add_argument("--testSynthetic", action="store_true")
    parser.add_argument("--testModels", action="store_true")
    parser.add_argument("--testRouter", action="store_true")
    parser.add_argument("--FINDER_type", type=str, default="FINDER",
                       help="FINDER type")

    args = parser.parse_args()
    FINDER_type = args.FINDER_type

    if args.testSynthetic:
        run_synthetic_evaluation(FINDER_type)

    if args.testReal:
        run_real_evaluation(FINDER_type)
    
    if args.testModels:
        run_models_evaluation(FINDER_type)

    if args.testRouter:
        run_router_evaluation(FINDER_type)