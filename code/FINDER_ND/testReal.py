#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
from unittest import result
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import json
import argparse

from testUtils import load_config, create_model, find_data_file, load_graph_from_file


def GetSolution(dqn,stepRatio, datasets, dataset_dir, save_result_dir):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    
    ## begin computing...
    sol_time_df = pd.DataFrame(np.arange(1*len(datasets)).reshape((1,len(datasets))),index=['time'], columns=datasets)

    for j in range(len(datasets)):
        print ('\nTesting dataset %s'%datasets[j])
        
        data_file = find_data_file(datasets[j], dataset_dir)
        if data_file is None:
            print(f"Warning: Could not find data file for {datasets[j]}")
            continue

        g_test = load_graph_from_file(data_file)
        if g_test is None:
            print(f"Warning: Could not load graph for {datasets[j]}")
            continue
        result_file = save_result_dir + datasets[j] + '.txt'
        solution, time = dqn.EvaluateRealData(g_test, result_file, stepRatio)
        sol_time_df.iloc[0,j] = time
        print('Data:%s, get_sol_time:%.2f'%(datasets[j], time))
    sol_time_df.to_csv(save_result_dir + 'solution_time.csv' , encoding='utf-8', index=False)

    

def EvaluateSolution(dqn,strategyID,datasets, dataset_dir, save_result_dir ):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
   
    ## begin computing...
    score_df = pd.DataFrame(np.arange(1 * len(datasets)).reshape((1, len(datasets))), index=['solution'], columns=datasets)
    for i in range(len(datasets)):
        print('\nEvaluating dataset %s' % datasets[i])
        data_file = find_data_file(datasets[i], dataset_dir)
        if data_file is None:
            print(f"Warning: Could not find data file for {datasets[i]}")
            continue
        g_test = load_graph_from_file(data_file)
        if g_test is None:
            print(f"Warning: Could not load graph for {datasets[i]}")
            continue
        solution = save_result_dir + datasets[i] + '.txt'
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        ################################## modify to choose which strategy to evaluate
        score, MaxCCList = dqn.EvaluateSol(g_test, solution, strategyID, reInsertStep=0.001)
        t2 = time.time()
        print('Data: %s, score:%.6f, eval_sol_time: %.6f'% (datasets[i], score,t2 - t1))

        score_df.iloc[0, i] = score
        result_file = save_result_dir + 'MaxCCList_Strategy_' + datasets[i] + '.txt'
        with open(result_file, 'w') as f_out:
            for j in range(len(MaxCCList)):
                f_out.write('%.8f\n' % MaxCCList[j])
    score_df.to_csv(save_result_dir + 'solution_score.csv', encoding='utf-8', index=False)

# def RandomRemoveEvaluate(dqn,STEPRATIO, REPEAT, MODEL_FILE_CKPT=None, datasets=None, dataset_dir=None, model_dir=None):

#     # randomRatioList = [0.005,0.01,0.02,0.05,0.1,0.15,0.2]
#     randomRatioList = [0.5]
#     ######################################################################################################################
#     ##................................................Get Solution (model).....................................................
#     # dqn = GraphDQN()
#     cfd = os.path.dirname(__file__)
#     ## data_set
#     data_test_path = dataset_dir or '%s/../../data/real/'%(cfd)
#     # datasets = ['Crime','HI-II-14']
#     ## model_file
#     model_file_path = model_dir or f'./models/'
#     if not MODEL_FILE_CKPT:
#         model_file_ckpt = dqn.findModel()
#     else:
#         model_file_ckpt = MODEL_FILE_CKPT
#     model_file = model_file_path + model_file_ckpt
#     ## modify to choose which stepRatio to get the solution
#     stepRatio = STEPRATIO
#     ## save_dir : save sol
#     save_dir = '%s/result/real'%(cfd)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     ## save_dir_local : save random test result
#     save_dir_local = save_dir + '/StepRatio_%.4f/evaluate_random' % stepRatio
#     if not os.path.exists(save_dir_local):
#         os.mkdir(save_dir_local)
        
    # ## begin computing...
    # print("------------------ random evaluate ------------------")
    # print('The best model is :%s'%(model_file))

    # for j in range(len(datasets)):
    #     print ('\nTesting dataset %s'%datasets[j])
    #     data_test = data_test_path + datasets[j] + '.txt'
    #     df = pd.DataFrame(np.arange(REPEAT*len(randomRatioList)).reshape((REPEAT,len(randomRatioList))), columns=randomRatioList)
    #     for r_i,r in enumerate(randomRatioList):
    #         for t in range(REPEAT):
    #             score, MaxCCList,solution, time = dqn.EvaluateRealData_random(model_file, data_test, save_dir, r, stepRatio)
    #             print("score:",score)
    #             # print("solution",solution)
    #             df.iloc[t,r_i] = score
    #     print('Data:%s, remove ratio:%f, time:%.2f'%(datasets[j], r, time))
    #     df.to_csv(save_dir_local + '/%s.csv'%(datasets[j]), encoding='utf-8', index=False)


def main():
    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # parse args
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--eval_all_iter", action="store_true")
    parser.add_argument("--test_iter", type=int, default=1200)
    parser.add_argument("--test_logic", action="store_true", help="Test evaluation logic without running evaluation")
    args = parser.parse_args()
    
    
    # Check if test_logic is enabled
    if args.test_logic:
        print("\nTesting evaluation logic...")
        try:
            from testUtils import test_evaluation_logic
            needed_iters, needed_datasets_per_iter = test_evaluation_logic(config)
            print(f"\nTest completed. {len(needed_iters)} iterations need evaluation.")
            return
        except Exception as e:
            print(f"✗ Error during test: {e}")
            return
    
    # Check if eval_all_iter is enabled
    if args.eval_all_iter == True:
        print("\neval_all_iter is True, running comprehensive evaluation...")
        # Import and run the comprehensive evaluation
        try:
            from testUtils import eval_all_iters,save_results,load_validation_scores,create_comparison_plot
            all_results = eval_all_iters(config)
    
            # Save results
            print("\nSaving results...")
            score_df, time_df = save_results(all_results, config)
    
            # Create comparison plot
            val_scores = load_validation_scores(config)
            create_comparison_plot(score_df, val_scores, config)
    
            print("\n✓ Evaluation completed successfully!")
            return
        except ImportError:
            print("Warning: eval_all_iterations.py not found, falling back to single iteration evaluation")
    
    else:
        print("\neval_all_iter is False, evaluate iteration %d"%args.test_iter)
        from testUtils import eval_one_iter,save_results
        
        results = []
        results.append(eval_one_iter(args.test_iter, config, save_sol= True))
        save_results(results, config)


    # # Create GraphDQN model from configuration
    # model_config = config['model_config']
    # test_iter = config['eval_config']['test_iter']
    # dqn = create_model(model_config,test_iter)
    
    # # Extract parameters from config
    # eval_config = config['eval_config']
    # data_config = config['data_config']
    
    # datasets = eval_config['datasets']
    # STEPRATIO = eval_config['step_ratio']
    # STRTEGYID = eval_config['strategy_id']
    # save_result_dir = eval_config['save_result_dir']
    # dataset_dir = data_config['dataset_dir']

    # save_dir_local = save_result_dir + '/%s/%s_%d_%d_StepRatio_%.4f/' %(dqn.g_type,dqn.embeddingMethod,dqn.num_min,dqn.num_max,STEPRATIO)
    # if not os.path.exists(save_dir_local):#make dir
    #         os.makedirs(save_dir_local)

    # # case 1
    # # RandomRemoveEvaluate(dqn,0.01,100,model_file)
    
    # # case 2
    # GetSolution(dqn,STEPRATIO, datasets, dataset_dir, save_dir_local,)
    # EvaluateSolution(dqn,STRTEGYID, datasets, dataset_dir, save_dir_local)

if __name__=="__main__":
    main()
