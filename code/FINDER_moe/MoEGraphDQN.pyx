#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
MoE GraphDQN - Mixture of Experts version for network dismantling
Based on the original GraphDQN but with MoE architecture
"""

from __future__ import print_function, division

# Suppress warnings and verbose output
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from cython.operator cimport dereference as deref
import tensorflow as tf
import tensorflow.compat.v1 as tf1

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
tf1.logging.set_verbosity(tf1.logging.ERROR)

import numpy as np
import networkx as nx
import random
import time
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import pickle
from MoEGraphDQN_modules import MoEGraphEncoder, MoEMLPDecoder

# Hyper Parameters (same as original)
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 1000000      #1,000,000 orgin,every 5000 generate new graphs
cdef double LEARNING_RATE = 0.0001   #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.0001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30
cdef int NUM_MAX = 120
cdef int REG_HIDDEN = 32
cdef int BATCH_SIZE = 64  
cdef double initialization_stddev = 0.01  # weight initialization standard deviation
cdef int n_valid = 200
cdef int aux_dim = 4
cdef int num_env = 1
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3  # embed layers num
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage

class MoEGraphDQN:

    def __init__(self,
        g_type = 'BA',
        g_params = {'nrange': '30_50',
                    'm':2},
        target_graph = "Digg",
        save_model_dir = './models',
        moe_config = {
            'num_experts': 4,
            'top_k': 2,
            'router_dropout': 0.1,
            'load_balance_loss_weight': 0.01
        }
    ):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = g_type #BA(barabasi_albert),ER(),PL(powerlaw), SW(small-world), ego
        self.g_params = g_params
        self.target_graph = target_graph
        self.num_min = int(g_params['nrange'].split('_')[0])
        self.num_max = int(g_params['nrange'].split('_')[1])
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()
        
        # MoE specific parameters
        self.moe_config = moe_config
        self.num_experts = moe_config['num_experts']
        self.top_k = moe_config['top_k']

        ############----------------------------- paths ------------------- ###################################
        self.train_dir = f"../../dataset/synthetic/GSDM"
        self.valid_dir = f"../../dataset/synthetic/GSDM"
        # train ego graph id,begin with 0
        self.dataset_id = 24    
        # save_model_dir: directory to save the models
        self.save_model_dir = f"{save_model_dir}/MoE_{self.g_type}_nrange_{g_params['nrange']}_m_{g_params['m']}"
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        # VCFile: file to store the validation results
        self.VCFile = os.path.join(self.save_model_dir, f"MoEModelVC.csv")

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False

        ############----------------------------- variants of DQN(end) ------------------- ###################################
        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)
        tf1.disable_eager_execution()
        
        # Placeholders (same as original)
        # [batch_size, node_cnt]
        self.action_select = tf1.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf1.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf1.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf1.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf1.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf1.placeholder(tf.float32, [None, 1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf1.placeholder(tf.float32, [None, aux_dim], name="aux_input")
        # [N] batch graph ids for routing
        self.batch_graph_ids = tf1.placeholder(tf.int32, [None], name="batch_graph_ids")
        
        # MoE specific placeholders
        # [batch_size, num_experts] - expert utilization for load balancing
        self.expert_utilization = tf1.placeholder(tf.float32, [None, self.num_experts], name="expert_utilization")

        # Build the MoE network
        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.trainable_vars = self.BuildMoENet()
        
        # Initialize session
        self.sess = tf1.Session()
        self.sess.run(tf1.global_variables_initializer())
        
        # Saver for model checkpointing
        self.saver = tf1.train.Saver(max_to_keep=10)

    def BuildMoENet(self):
        """Build the MoE network architecture"""
        # N: number of nodes (of all graphs in a batch)
        nodes_size = tf.shape(self.n2nsum_param)[0]
        # B: batch_size (number of graphs in a batch)
        y_nodes_size = tf.shape(self.subgsum_param)[0]

        feature_size = 2 # 2 features for node and graph

        # X: [N,feature_size] node feature, initialized with 1
        node_input = tf.cast(tf.ones((nodes_size,feature_size)),tf.float32)
        # Y: [B,feature_size] graph feature, initialized with 1
        y_node_input = tf.cast(tf.ones((y_nodes_size,feature_size)),tf.float32)

        # MoE Encoder
        encoder = MoEGraphEncoder(
            embedding_size=self.embedding_size,
            feature_size=feature_size,
            initialization_stddev=initialization_stddev,
            gnn_layers=max_bp_iter
        )
        
        # MoE Decoder
        decoder = MoEMLPDecoder(
            embedding_size=self.embedding_size,
            reg_hidden=self.reg_hidden,
            aux_dim=aux_dim,
            initialization_stddev=initialization_stddev
        )

        # Encoder: get node and graph embeddings with expert weights
        cur_message_layer, y_cur_message_layer, expert_weights = encoder.encode(
            node_input,
            y_node_input,
            self.n2nsum_param, 
            self.subgsum_param,
            self.batch_graph_ids
        )
        
        # Decoder: get Q(s,a) [B, 1]
        q_pred = decoder.decode_q(
            cur_message_layer,
            self.action_select,
            y_cur_message_layer, 
            self.aux_input
        )

        # Decoder: get Q(s,a) for all nodes of all graphs [N, 1]
        q_on_all = decoder.decode_q_all(
            cur_message_layer,
            y_cur_message_layer, 
            self.aux_input,
            self.rep_global
        )

        # Reconstruction loss (same as original)
        loss_recons = 2 * tf1.trace(tf.matmul(tf.transpose(cur_message_layer), tf1.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param, tf.float32), cur_message_layer)))
        edge_num = tf1.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))
        loss_recons = tf.divide(loss_recons, edge_num)
        
        # RL loss
        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)
        
        # MoE Load Balancing Loss
        # Encourage equal utilization of experts
        expert_utilization_mean = tf.reduce_mean(expert_weights, axis=0)  # [num_experts]
        target_utilization = 1.0 / tf.cast(self.num_experts, tf.float32)
        load_balance_loss = tf.reduce_mean(tf.squared_difference(expert_utilization_mean, target_utilization))
        
        # Total loss
        loss = loss_rl + Alpha * loss_recons + self.moe_config['load_balance_loss_weight'] * load_balance_loss
        
        trainStep = tf1.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return loss, trainStep, q_pred, q_on_all, tf1.trainable_variables()

    def gen_graph(self, num_min, num_max):
        """Generate a single graph (same as original)"""
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'ER':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'PL':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'SW':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'BA':
            g = nx.barabasi_albert_graph(n=cur_n, m=self.g_params['m'])
        elif self.g_type == 'ego':
            print("please implement one ego graph generation")
            exit()
        return g

    def gen_new_graphs(self, num_min, num_max):
        """Generate new training graphs (same as original)"""
        print('Generating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        if self.g_type in ['ER','PL','SW','BA']:
            for i in tqdm(range(1000), desc="Training graphs"):
                g = self.gen_graph(num_min, num_max)
                self.InsertGraph(g, is_test=False)
        elif self.g_type in ['ego']:
            graphs = pickle.load(open(f"{self.train_dir}/{self.target_graph}_ego_train_{self.dataset_id}.pkl", 'rb'))
            print(f"Loading training graphs from {self.train_dir} (id: {self.dataset_id})")
            self.dataset_id += 1
            for i in tqdm(range(1000), desc="Training graphs"):
                g = graphs[i]
                self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        """Clear training graphs (same as original)"""
        self.TrainSet.Clear()
        self.ngraph_train = 0

    def InsertGraph(self, g, is_test=False):
        """Insert a graph into the dataset (same as original)"""
        if is_test:
            self.TestSet.InsertGraph(g)
            self.ngraph_test += 1
        else:
            self.TrainSet.InsertGraph(g)
            self.ngraph_train += 1

    def PrepareValidData(self):
        """Prepare validation data (same as original)"""
        print('Preparing validation data...')
        sys.stdout.flush()
        self.ClearTestGraphs()
        if self.g_type in ['ER','PL','SW','BA']:
            for i in tqdm(range(200), desc="Validation graphs"):
                g = self.gen_graph(self.num_min, self.num_max)
                self.InsertGraph(g, is_test=True)
        elif self.g_type in ['ego']:
            graphs = pickle.load(open(f"{self.valid_dir}/{self.target_graph}_ego_test_{self.dataset_id}.pkl", 'rb'))
            for i in tqdm(range(200), desc="Validation graphs"):
                g = graphs[i]
                self.InsertGraph(g, is_test=True)

    def ClearTestGraphs(self):
        """Clear test graphs (same as original)"""
        self.TestSet.Clear()
        self.ngraph_test = 0

    def PlayGame(self, int n_traj, double eps):
        """Play games to collect experience (same as original)"""
        cdef int i, j
        for i in range(n_traj):
            self.env_list[0].s0(self.TrainSet.Get(i % self.ngraph_train))
            self.g_list[0] = self.env_list[0].graph
            while (not self.env_list[0].isTerminal()):
                list_pred = self.PredictWithCurrentQNet([self.g_list[0]], [self.env_list[0].action_list])
                if random.random() < eps:
                    action = random.choice(self.env_list[0].action_list)
                else:
                    action = self.argMax(list_pred[0])
                self.env_list[0].step(action)

    def PredictWithCurrentQNet(self, g_list, action_list):
        """Predict Q-values with current network (adapted for MoE)"""
        # Prepare batch data
        batch_data = PrepareBatchGraph.py_PrepareBatchGraph(g_list, action_list)
        
        # Get expert utilization (initialize with uniform distribution)
        batch_size = len(g_list)
        expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
        
        # Run prediction
        q_values = self.sess.run(self.q_on_all, feed_dict={
            self.n2nsum_param: batch_data['n2nsum_param'],
            self.subgsum_param: batch_data['subgsum_param'],
            self.rep_global: batch_data['rep_global'],
            self.aux_input: batch_data['aux_input'],
            self.batch_graph_ids: batch_data['batch_graph_ids'],
            self.expert_utilization: expert_util
        })
        
        return q_values

    def argMax(self, list_pred):
        """Get action with maximum Q-value (same as original)"""
        cdef int i
        cdef double max_val = -inf
        cdef int max_idx = 0
        for i in range(len(list_pred)):
            if list_pred[i] > max_val:
                max_val = list_pred[i]
                max_idx = i
        return max_idx

    def Fit(self):
        """Fit the model (adapted for MoE)"""
        if self.nStepReplayMem.size() < BATCH_SIZE:
            return
        
        # Sample batch
        if self.IsPrioritizedSampling:
            batch_data, batch_ISWeights = self.nStepReplayMem.sample(BATCH_SIZE)
            self.ISWeights = batch_ISWeights
        else:
            batch_data = self.nStepReplayMem.sample(BATCH_SIZE)
        
        # Get expert utilization for load balancing
        batch_size = len(batch_data['g_list'])
        expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
        
        # Train
        self.sess.run(self.trainStep, feed_dict={
            self.n2nsum_param: batch_data['n2nsum_param'],
            self.subgsum_param: batch_data['subgsum_param'],
            self.action_select: batch_data['action_select'],
            self.rep_global: batch_data['rep_global'],
            self.target: batch_data['target'],
            self.aux_input: batch_data['aux_input'],
            self.batch_graph_ids: batch_data['batch_graph_ids'],
            self.expert_utilization: expert_util
        })

    def TakeSnapShot(self):
        """Take snapshot of target network (same as original)"""
        # This would be implemented if using target networks
        pass

    def SaveModel(self, filename):
        """Save model checkpoint"""
        save_path = os.path.join(self.save_model_dir, filename)
        self.saver.save(self.sess, save_path)
        print(f"Model saved to {save_path}")

    def LoadModel(self, filename):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_model_dir, filename)
        self.saver.restore(self.sess, load_path)
        print(f"Model loaded from {load_path}")

    def resume_checkpoint_and_iter(self):
        """Resume from latest checkpoint"""
        if not os.path.exists(self.save_model_dir):
            return None, 0
        
        # Find latest checkpoint
        ckpt_files = [f for f in os.listdir(self.save_model_dir) if f.endswith('.ckpt.index')]
        if not ckpt_files:
            return None, 0
        
        # Extract iteration number from filename
        latest_ckpt = sorted(ckpt_files)[-1]
        ckpt_name = latest_ckpt.replace('.ckpt.index', '')
        
        # Extract iteration number
        try:
            iter_num = int(ckpt_name.split('_')[-1])
        except:
            iter_num = 0
        
        return ckpt_name, iter_num

    def resume_nlines_and_runtime(self):
        """Resume runtime from CSV file"""
        if not os.path.exists(self.VCFile):
            return 0, 0
        
        with open(self.VCFile, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= 1:  # Only header or empty
            return 0, 0
        
        # Get runtime from last line
        last_line = lines[-1]
        try:
            parts = last_line.strip().split(',')
            runtime = float(parts[2]) if len(parts) > 2 else 0
        except:
            runtime = 0
        
        return len(lines) - 1, runtime

    def Train(self):
        """Main training loop (adapted for MoE)"""
        self.PrepareValidData()
        self.gen_new_graphs(self.num_min, self.num_max) # gen 1000 graphs for training

        cdef int i, iter, idx
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        cdef double eps_start = 1.0
        cdef double eps_end = 0.05
        cdef double eps_step = 10000.0
        cdef int loss = 0
        cdef double frac, start, end
        
        start_iter = 0
        last_ckpt, last_iter = self.resume_checkpoint_and_iter()
        if last_ckpt != None:
            print(f"Resuming from checkpoint: {last_ckpt} at iter {last_iter}")
            self.LoadModel(last_ckpt)
            start_iter = last_iter + 1
            _,runtime = self.resume_nlines_and_runtime()
            # Open CSV in append mode
            f_out = open(self.VCFile, 'a')
        else:
            print("\nNo checkpoint found, starting from scratch.")
            # Start from scratch, overwrite CSV
            start_iter = 0
            runtime = 0
            f_out = open(self.VCFile, 'w')

        t_train_start = time.time()
        for iter in range(start_iter, MAX_ITERATION):
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(self.num_min, self.num_max)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter == start_iter:
                N_start = time.perf_counter()
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                frac = 0.0 
                test_start = time.time()
                ########--------- Test --------########
                for idx in range(n_valid):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%d, %.8f, %.4f\n'%(iter,frac/n_valid, test_end-t_train_start+runtime))   #write vc into the file
                f_out.flush()
                print('iter %d, eps %.4f, average size of vc:%.6f'%(iter, eps, frac/n_valid))
                print ('testing 200 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.perf_counter()
                print('300 iterations time: %.2fs\n'%(N_end-N_start))
                sys.stdout.flush()
                ########--------- Save --------########
                ckpt_file = 'MoE_iter_%d.ckpt' % iter
                self.SaveModel(ckpt_file)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            ########--------- Fit --------########    
            self.Fit()
        f_out.close()

    def Test(self,int gid):
        """Test when train (same as original)"""
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        cdef int i
        sol = []
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness

    def findModel(self):
        """Find best model by validation score (same as original)"""
        vc_list = []
        for line in open(self.VCFile):
            line=line.split(",")[1]
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        best_model = 'MoE_iter_%d.ckpt' % best_model_iter
        print("Finding best model by validation score: %s"%best_model)
        return best_model

    def Evaluate(self, test_graphs):
        """Evaluate on a number of graphs (same as original)"""
        sys.stdout.flush()
        g_num = len(test_graphs)
        # Implementation would be similar to original
        pass
