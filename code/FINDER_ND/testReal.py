#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import json

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_model(model_config):
    """Create GraphDQN model from configuration"""
    gnn_model = model_config['gnn_model']
    num_min = model_config['num_min']
    num_max = model_config['num_max']
    iter = model_config['iter']
    # e.g. GIN_nrange_30_50_iter_2700.ckpt
    ckpt_file = f"{gnn_model}_nrange_{num_min}_{num_max}_iter_{iter}.ckpt"
    
    dqn = GraphDQN(
        g_type=model_config['g_type'],
        gnn_model=model_config['gnn_model'],
        target_graph=model_config['target_graph'],
        num_min=model_config['num_min'],
        num_max=model_config['num_max'],
        ckpt_file = ckpt_file
    )
    return dqn

def find_data_file(dataset_name, dataset_dir):
    if dataset_name in ['Crime','HI-II-14','Digg','Enron','Gnutella31','Facebook','Epinions','Youtube','Flickr']:
        data_file = f'{dataset_dir}/%s.txt'%(dataset_name)
    elif dataset_name in ['corruption']:
        data_file = f'{dataset_dir}/%s.gt'%(dataset_name)
    else: 
        data_file = None
    
    return data_file


def load_graph_from_file(data_file):
    if data_file.split('.')[-1] == 'txt': 
        G = nx.read_edgelist(data_file,nodetype=int)
    else:
        try:
            import graph_tool.all as gt
            gt_graph = gt.load_graph(data_file)
            G = nx.Graph()
            for e in gt_graph.edges():
                u = int(e.source())
                v = int(e.target())
                G.add_edge(u, v)
        except ImportError:
            print("Warning: graph_tool not available, skipping .gt files")
            return None
    return G

def GetSolution(stepRatio, save_result_dir, dqn, datasets, dataset_dir):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    
    ## begin computing...
    sol_time_df = pd.DataFrame(np.arange(1*len(datasets)).reshape((1,len(datasets))),index=['time'], columns=datasets)

    for j in range(len(datasets)):
        print ('\nTesting dataset %s'%datasets[j])
        data_test_file = find_data_file(datasets[j], dataset_dir)
        if data_test_file is None:
            print(f"Warning: Could not find data file for {datasets[j]}")
            continue
        g_test = load_graph_from_file(data_test_file)
        if g_test is None:
            print(f"Warning: Could not load graph for {datasets[j]}")
            continue
        result_file = save_result_dir + datasets[j] + '.txt'
        solution, time = dqn.EvaluateRealData(g_test, result_file, stepRatio)
        sol_time_df.iloc[0,j] = time
        print('Data:%s, get_sol_time:%.2f'%(datasets[j], time))
    sol_time_df.to_csv(save_result_dir + 'solution_time.csv' , encoding='utf-8', index=False)

    

def EvaluateSolution(strategyID, save_result_dir, dqn, datasets, dataset_dir):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
   
    ## begin computing...
    score_df = pd.DataFrame(np.arange(1 * len(datasets)).reshape((1, len(datasets))), index=['solution'], columns=datasets)
    for i in range(len(datasets)):
        print('\nEvaluating dataset %s' % datasets[i])
        data_test_file = find_data_file(datasets[i], dataset_dir)
        if data_test_file is None:
            print(f"Warning: Could not find data file for {datasets[i]}")
            continue
        g_test = load_graph_from_file(data_test_file)
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

def RandomRemoveEvaluate(STEPRATIO, REPEAT, MODEL_FILE_CKPT=None, dqn=None, datasets=None, dataset_dir=None, model_dir=None):

    # randomRatioList = [0.005,0.01,0.02,0.05,0.1,0.15,0.2]
    randomRatioList = [0.5]
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    # dqn = GraphDQN()
    cfd = os.path.dirname(__file__)
    ## data_set
    data_test_path = dataset_dir or '%s/../../data/real/'%(cfd)
    # datasets = ['Crime','HI-II-14']
    ## model_file
    model_file_path = model_dir or './FINDER_ND/models/barabasi_albert/'
    if not MODEL_FILE_CKPT:
        model_file_ckpt = dqn.findModel()
    else:
        model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + model_file_ckpt
    ## modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    ## save_dir : save sol
    save_dir = '%s/result/real'%(cfd)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ## save_dir_local : save random test result
    save_dir_local = save_dir + '/StepRatio_%.4f/evaluate_random' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
        
    ## begin computing...
    print("------------------ random evaluate ------------------")
    print('The best model is :%s'%(model_file))

    for j in range(len(datasets)):
        print ('\nTesting dataset %s'%datasets[j])
        data_test = data_test_path + datasets[j] + '.txt'
        df = pd.DataFrame(np.arange(REPEAT*len(randomRatioList)).reshape((REPEAT,len(randomRatioList))), columns=randomRatioList)
        for r_i,r in enumerate(randomRatioList):
            for t in range(REPEAT):
                score, MaxCCList,solution, time = dqn.EvaluateRealData_random(model_file, data_test, save_dir, r, stepRatio)
                print("score:",score)
                # print("solution",solution)
                df.iloc[t,r_i] = score
        print('Data:%s, remove ratio:%f, time:%.2f'%(datasets[j], r, time))
        df.to_csv(save_dir_local + '/%s.csv'%(datasets[j]), encoding='utf-8', index=False)


def main():
    # Load configuration
    config = load_config()
    
    # Create GraphDQN model from configuration
    model_config = config['model_config']
    dqn = create_model(model_config)
    
    # Extract parameters from config
    eval_config = config['eval_config']
    data_config = config['data_config']
    
    datasets = eval_config['datasets']
    STEPRATIO = eval_config['step_ratio']
    STRTEGYID = eval_config['strategy_id']
    save_result_dir = eval_config['save_result_dir']
    dataset_dir = data_config['dataset_dir']
 

    save_dir_local = save_result_dir + '/%s/%s_%d_%d_StepRatio_%.4f/' %(dqn.g_type,dqn.embeddingMethod,dqn.num_min,dqn.num_max,STEPRATIO)
    if not os.path.exists(save_dir_local):#make dir
            os.makedirs(save_dir_local)


    # case 1
    # RandomRemoveEvaluate(0.01,100,model_file)
    
    # case 2
    GetSolution(STEPRATIO, save_dir_local, dqn, datasets, dataset_dir)
    EvaluateSolution(STRTEGYID, save_dir_local, dqn, datasets, dataset_dir)

if __name__=="__main__":
    main()
