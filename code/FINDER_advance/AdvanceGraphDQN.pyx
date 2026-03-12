#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
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
import json
from GraphDQN_modules import GraphEncoder, MLPDecoder
from MoEGraphDQN_modules import MoEGraphEncoder,MoEMLPDecoder

# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 1000000      #1,000,000 orgin,every 5000 generate new graphs
cdef double LEARNING_RATE = 0.0001   #increased for better learning
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
# cdef int NUM_MIN = 30
# cdef int NUM_MAX = 120
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


class AdvanceGraphDQN:

    def __init__(self,
        **model_config
    ):
        # init some parameters
        self.embeddingMethod = model_config["gnn_model"]
        
        self.g_type = model_config["g_type"] #BA(barabasi_albert),ER(),PL(powerlaw), SW(small-world), ego
        self.g_params = model_config["g_params"]
        self.num_min = int(model_config["g_params"]['nrange'].split('_')[0])
        self.num_max = int(model_config["g_params"]['nrange'].split('_')[1])
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.utils = utils.py_Utils()
        self.options = model_config.get("options",{})

        self.hyperparameters = model_config.get("hyperparameters",{})
        self.embedding_size = self.hyperparameters.get("embedding_size",EMBEDDING_SIZE)
        self.reg_hidden = self.hyperparameters.get("reg_hidden",REG_HIDDEN)
        self.learning_rate = self.hyperparameters.get("learning_rate",LEARNING_RATE)
        self.gamma = self.hyperparameters.get("gamma",GAMMA)
        self.batch_size = self.hyperparameters.get("batch_size",BATCH_SIZE)

        ############----------------------------- paths ------------------- ###################################
        # self.train_dir = f"../../dataset/synthetic/GSDM"
        # self.valid_dir = f"../../dataset/synthetic/GSDM"
        # train ego graph id,begin with 0
        # self.dataset_id = 24    
        # save_model_dir: directory to save the models
        save_model_dir =  model_config["save_model_dir"]
        if self.g_type == 'ego' or self.g_type == 'mix':
            self.save_model_dir = f"{save_model_dir}/{self.embeddingMethod}_{self.g_type}_nrange_{self.g_params['nrange']}"
        else: self.save_model_dir = f"{save_model_dir}/{self.embeddingMethod}_{self.g_type}_nrange_{self.g_params['nrange']}_m_{self.g_params['m']}"

        
        ############----------------------------- variants of DQN(start) ------------------- ###################################
        if self.embeddingMethod=='MoE':
            # MoE specific parameters
            self.moe_config = model_config.get("moe_config", {})
            if not self.moe_config:
                self.moe_config['top_k'] = 2
                self.moe_config['from_pretrained'] = False
                self.moe_config['pretrained_model_dir'] = None
                self.moe_config['num_experts'] = 4
        
        self.IsFeatures =  self.options.get('IsFeatures',False)
        if self.IsFeatures:
             self.save_model_dir += "_AF"
        
        self.IsDisturbG = self.options.get('IsDisturbG',False)
        if self.IsDisturbG:
             self.save_model_dir += "_DG"
             
        self.IsDoubleDQN = self.options.get('IsDoubleDQN',False)
        if self.IsDoubleDQN:
            self.save_model_dir += "_DDQN"
        
        self.IsPrioritizedSampling = self.options.get('IsPrioritizedSampling',False)
        if self.IsPrioritizedSampling:
             self.save_model_dir += "_Prioritized"

        self.IsHuberloss = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False

        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

        # save model config
        config = {}
        config["model_config"] = model_config
        config_path = os.path.join(self.save_model_dir,'config.json')
        with open(config_path,'w',encoding='utf-8') as f:
            json.dump(config, f,ensure_ascii=False,indent=4)
            print(f"save config in {config_path}")

        # VCFile: file to store the validation results
        self.VCFile = os.path.join(self.save_model_dir, f"ModelVC_{self.embeddingMethod}.csv")

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
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE,self.gamma)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(self.num_max))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(self.num_max)
        tf1.disable_eager_execution()
        # [batch_size, node_cnt]
        self.action_select = tf1.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf1.sparse_placeholder(tf.float32, name="rep_global")
        # adjacency matrix, [node_cnt, node_cnt] 
        self.n2nsum_param = tf1.sparse_placeholder(tf.float32, name="n2nsum_param")
        #  laplacian matrix, [node_cnt, node_cnt]
        self.laplacian_param = tf1.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt] 
        self.subgsum_param = tf1.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf1.placeholder(tf.float32, [self.batch_size,1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf1.placeholder(tf.float32, name="aux_input")
        # [N]
        self.batch_graph_ids = tf1.placeholder(tf.int32, name="batch_graph_ids")

        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf1.placeholder(tf.float32, [self.batch_size, 1], name='IS_weights')

        if self.embeddingMethod=='MoE':
            # Build the MoE network
            self.encoder, self.decoder, self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildMoENet()
            # init Target Q Network
            self.encoderT, self.decoderT,self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT = self.BuildMoENet(is_target = True)
        else:
            # init Q network
            self.encoder,self.decoder,self.loss,self.trainStep,self.q_pred, self.q_on_all,self.Q_param_list = self.BuildNet() #[loss,trainStep,q_pred, q_on_all, ...]
            # init Target Q Network
            self.encoderT,self.decoderT,self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT = self.BuildNet()
        
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)]


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf1.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf1.ConfigProto(device_count={"CPU": 6},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf1.Session(config = config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf1.global_variables_initializer())
        

        # w_n2l_val = self.session.run(tf1.get_default_graph().get_tensor_by_name('Variable:0'))
        # print(f"Encoder w_n2l stats - {w_n2l_val}")
        # cross_val = self.session.run(tf1.get_default_graph().get_tensor_by_name('Variable_6:0'))
        # print(f"Decoder cross_product stats - {cross_val}")
        # w_n2l_val = self.session.run(tf1.get_default_graph().get_tensor_by_name('Variable_7:0'))
        # print(f"EncoderT w_n2l stats -{w_n2l_val}")
        # cross_val = self.session.run(tf1.get_default_graph().get_tensor_by_name('Variable_13:0'))
        # print(f"DecoderT cross_product stats -{cross_val}")

        # print("after BuildNet")
        # w_n2l_val = self.session.run(self.encoder.w_n2l)
        # print(f"Encoder w_n2l stats - {w_n2l_val}")
        # cross_val = self.session.run(self.decoder.cross_product)
        # print(f"Decoder cross_product stats - {cross_val}")
        # w_n2l_val = self.session.run(self.encoderT.w_n2l)
        # print(f"EncoderT w_n2l stats -{w_n2l_val}")
        # cross_val = self.session.run(self.decoderT.cross_product)
        # print(f"DecoderT cross_product stats -{cross_val}")


        # Print GPU information (cleaner version)
        gpus = tf1.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {len(gpus)} device(s) available, Using device: {gpus[0].name}")
            print(f"tf.is_gpu_available() : {tf.test.is_gpu_available()}\n")
        else:
            print("[WARNING] No GPU detected - training will use CPU")
        

#################################################code for graphDQN#####################################
    def BuildNet(self):
        # N: number of nodes (of all graphs in a batch)
        nodes_size = tf.shape(self.n2nsum_param)[0]
        # B: batch_size (number of graphs in a batch)
        y_nodes_size = tf.shape(self.subgsum_param)[0]

        if self.IsFeatures:
            # Get feature strategy from options (default: simple_aggregate)
            feature_strategy = self.options.get('feature_strategy', 'simple_aggregate')
            # Options: 'simple_aggregate' or 'concat_baseline'
            feature_size = 4  # Both strategies use 4 features
            node_input, y_node_input = self.advanced_features(strategy=feature_strategy)
        else:
            feature_size = 2  # Simple features: initialized with 1
            # X: [N,feature_size] node feature, initialized with 1
            node_input = tf.cast(tf.ones((nodes_size,feature_size)),tf.float32)
            # Y: [B,feature_size] graph feature, initialized with 1
            y_node_input = tf.cast(tf.ones((y_nodes_size,feature_size)),tf.float32)

        encoder = GraphEncoder(
            embeddingMethod=self.embeddingMethod,
            feature_size = feature_size,
            embedding_size=self.embedding_size,
            initialization_stddev=initialization_stddev,
            gnn_layers=max_bp_iter
        )
        decoder = MLPDecoder(
            embedding_size=self.embedding_size,
            reg_hidden=self.reg_hidden,
            aux_dim=aux_dim,
            initialization_stddev=initialization_stddev
        )

        w_n2l_val = encoder.w_n2l
        print(f"BuildNet: Encoder w_n2l stats -{w_n2l_val}")
        cross_val = decoder.cross_product
        print(f"BuildNet: Decoder cross_product stats - {cross_val}")
        
        # Encoder: get node and graph embeddings
        cur_message_layer, y_cur_message_layer= encoder.encode(
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

        # Reconstruction loss
        loss_recons = 2 * tf1.trace(tf.matmul(tf.transpose(cur_message_layer), tf1.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param, tf.float32), cur_message_layer)))
        edge_num = tf1.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))
        loss_recons = tf.divide(loss_recons, edge_num)
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
        loss = loss_rl + Alpha * loss_recons
        trainStep = tf1.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return encoder,decoder,loss, trainStep, q_pred, q_on_all, tf1.trainable_variables()


    def BuildMoENet(self,is_target=False):
        """Build the MoE network architecture - CREATES SEPARATE INSTANCES"""
        # N: number of nodes (of all graphs in a batch)
        nodes_size = tf.shape(self.n2nsum_param)[0]
        # B: batch_size (number of graphs in a batch)
        y_nodes_size = tf.shape(self.subgsum_param)[0]

        if self.IsFeatures:
            # Get feature strategy from options (default: simple_aggregate)
            feature_strategy = self.options.get('feature_strategy', 'simple_aggregate')
            # Options: 'simple_aggregate' or 'concat_baseline'
            feature_size = 4  # Both strategies use 4 features
            node_input, y_node_input = self.advanced_features(strategy=feature_strategy)
        else:
            feature_size = 2  # Simple features: initialized with 1
            # X: [N,feature_size] node feature, initialized with 1
            node_input = tf.cast(tf.ones((nodes_size,feature_size)),tf.float32)
            # Y: [B,feature_size] graph feature, initialized with 1
            y_node_input = tf.cast(tf.ones((y_nodes_size,feature_size)),tf.float32)

        # CRITICAL FIX: Create separate instances with unique scopes
        # Generate unique scope suffix for each call (main vs target network)
        if is_target:
            scope_suffix = f"T"
        else: scope_suffix = ""
        
        with tf1.variable_scope(f"MoEEncoder{scope_suffix}", reuse=False):
            encoder = MoEGraphEncoder(
                embedding_size=self.embedding_size,
                feature_size=feature_size,
                initialization_stddev=initialization_stddev,
                gnn_layers=max_bp_iter,
                moe_config=self.moe_config
            )
        
        # Update num_experts only once (from first encoder)
        if not hasattr(self, 'num_experts'):
            self.num_experts = encoder.num_experts
            print(f"MoE initialized with {self.num_experts} experts")
        
        # CRITICAL FIX: Create decoder with unique scope
        with tf1.variable_scope(f"MoEDecoder{scope_suffix}", reuse=False):
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
                loss_rl = tf.reduce_mean(self.ISWeights * tf1.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)
        
        # MoE Load Balancing Loss (TEMPORARILY DISABLED FOR DEBUGGING)
        # Since we're using uniform expert weights, load balancing loss is unnecessary
        # and might interfere with training
        # expert_utilization_mean = tf.reduce_mean(expert_weights, axis=0)  # [num_experts]
        # target_utilization = 1.0 / tf.cast(self.num_experts, tf.float32)
        # load_balance_loss = tf.reduce_mean(tf1.squared_difference(expert_utilization_mean, target_utilization))
        load_balance_loss = 0.0  # Disable load balancing loss
        
        # Total loss (without load balancing for now)
        loss = loss_rl + Alpha * loss_recons  # + self.moe_config['load_balance_loss_weight'] * load_balance_loss
        
        trainStep = tf1.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return encoder,decoder,loss, trainStep, q_pred, q_on_all, tf1.trainable_variables()

    def advanced_features(self, strategy='simple_aggregate'):
        """
        Compute advanced structural features for nodes and graphs
        
        Args:
            strategy: 'simple_aggregate' or 'concat_baseline'
                - 'simple_aggregate': Advanced features only, aggregate to graph level
                - 'concat_baseline': Concatenate baseline + advanced features
        
        Returns:
            node_input: [N, feature_size] node features
            y_node_input: [B, feature_size] graph-level features
        """
        # Get dimensions
        N = tf.shape(self.n2nsum_param)[0]  # total nodes across batch
        B = tf.shape(self.subgsum_param)[0]  # batch size
        
        # ===== Compute Advanced Node-Level Features =====
        
        # First compute degree (needed for multiple features)
        degree = tf.sparse.reduce_sum(self.n2nsum_param, axis=1)  # [N]
        degree = tf.expand_dims(degree, axis=1)  # [N, 1]
        
        # Graph size for each graph (to normalize features)
        graph_sizes = tf.sparse.reduce_sum(self.subgsum_param, axis=1)  # [B]
        graph_sizes = tf.expand_dims(graph_sizes, axis=1)  # [B, 1]
        
        # Broadcast graph sizes to each node
        node_graph_sizes = tf1.sparse_tensor_dense_matmul(
            tf.sparse.transpose(tf.cast(self.subgsum_param, tf.float32)),
            graph_sizes
        )  # [N, 1]
        
        # 1. Estimated Clustering Coefficient
        # Sum of neighbor degrees (used for triangle approximation)
        neighbor_degrees = tf1.sparse_tensor_dense_matmul(
            tf.cast(self.n2nsum_param, tf.float32),
            degree  # [N, 1]
        )  # [N, 1] - sum of degrees of neighbors
        
        # Approximate number of triangles: neighbor_degrees / 2
        approx_triangles = neighbor_degrees / 2.0  # [N, 1]
        
        # Maximum possible triangles for each node: degree * (degree - 1) / 2
        max_triangles = degree * (degree - 1.0) / 2.0  # [N, 1]
        
        # Clustering coefficient: triangles / max_triangles
        clustering_coef = approx_triangles / (max_triangles + 1e-8)  # [N, 1]
        clustering_coef = tf.clip_by_value(clustering_coef, 0.0, 1.0)  # [N, 1]
        
        # 2. Normalized degree (degree / graph_size)
        norm_degree = degree / (node_graph_sizes + 1e-8)  # [N, 1]
        
        # 3. 2-hop neighbor density
        twohop_neighbors = neighbor_degrees / (node_graph_sizes * node_graph_sizes + 1e-8)  # [N, 1]
        
        # 4. Bias term
        node_bias = tf.ones([N, 1], dtype=tf.float32)  # [N, 1]
        
        # ===== Strategy 1: Simple Aggregate (Advanced features only) =====
        if strategy == 'simple_aggregate':
            # Node features: advanced structural features
            node_input = tf.concat([
                clustering_coef,     # [N, 1]
                norm_degree,         # [N, 1]
                twohop_neighbors,    # [N, 1]
                node_bias           # [N, 1]
            ], axis=1)  # [N, 4]
            
            # Graph features: aggregate node features (mean pooling)
            # This ensures y_node_input has same semantic meaning as node_input
            graph_node_count = tf.expand_dims(tf.sparse.reduce_sum(self.subgsum_param, axis=1), axis=1)  # [B, 1]
            y_node_input = tf1.sparse_tensor_dense_matmul(
                tf.cast(self.subgsum_param, tf.float32), 
                node_input
            ) / (graph_node_count + 1e-8)  # [B, 4] - mean of node features per graph
            
        # ===== Strategy 2: Concatenate with Baseline =====
        elif strategy == 'concat_baseline':
            # Baseline features (uniform initialization)
            node_baseline = tf.ones([N, 2], dtype=tf.float32)  # [N, 2]
            graph_baseline = tf.ones([B, 2], dtype=tf.float32)  # [B, 2]
            
            # Node features: baseline + advanced (keep only 2 most important advanced features)
            node_input = tf.concat([
                node_baseline,       # [N, 2] - baseline uniform features
                norm_degree,         # [N, 1] - most important: connectivity
                clustering_coef     # [N, 1] - second: local structure
            ], axis=1)  # [N, 4]
            
            # Graph features: baseline + aggregated advanced features
            graph_node_count = tf.expand_dims(tf.sparse.reduce_sum(self.subgsum_param, axis=1), axis=1)  # [B, 1]
            graph_norm_degree = tf1.sparse_tensor_dense_matmul(
                tf.cast(self.subgsum_param, tf.float32), 
                norm_degree
            ) / (graph_node_count + 1e-8)  # [B, 1]
            
            graph_clustering = tf1.sparse_tensor_dense_matmul(
                tf.cast(self.subgsum_param, tf.float32), 
                clustering_coef
            ) / (graph_node_count + 1e-8)  # [B, 1]
            
            y_node_input = tf.concat([
                graph_baseline,      # [B, 2] - baseline uniform features
                graph_norm_degree,   # [B, 1] - average degree
                graph_clustering    # [B, 1] - average clustering
            ], axis=1)  # [B, 4]
        
        elif strategy == 'original':
            # Concatenate all node features
            node_input = tf.concat([
                clustering_coef,     # [N, 1] - estimated clustering coefficient
                norm_degree,         # [N, 1] - normalized degree
                twohop_neighbors,    # [N, 1] - 2-hop neighbor density
                node_bias           # [N, 1] - bias term
            ], axis=1)  # [N, 4]
            
            # ===== Graph-Level Features =====
            # Extract from aux_input (already computed in C++)
            # aux_input shape: [B, 4]
            # aux_input contains: [covered_node_ratio, covered_edge_ratio, twohop_density, 1.0]
            y_node_input = self.aux_input  # [B, 4]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return node_input, y_node_input
    
    def gen_graph(self,cur_n):     
        # Select graph type (for mix, randomly choose)
        if self.g_type == 'mix':
            graph_types = ['BA', 'ER', 'PL', 'SW']
            # Get mix weights from config, default to uniform distribution
            weights = self.g_params.get('mix_weights', [0.2, 0.2, 0.2, 0.2])
            selected_type = np.random.choice(graph_types, p=weights)
        else:
            selected_type = self.g_type
        
        # Generate graph based on selected type
        seed = np.random.randint(1,1000)
        if selected_type == 'ER':
            g = nx.erdos_renyi_graph(n=cur_n, 
                                    p=self.g_params.get('p',random.choice([0.05,0.15,0.25,0.35])),
                                    seed=seed)
        elif selected_type == 'PL':
            g = nx.powerlaw_cluster_graph(n=cur_n, 
                                            m=self.g_params.get('m', random.choice([3,4,5,6])), 
                                            p=self.g_params.get('p', random.choice([0.05,0.15,0.25,0.35])),
                                            seed=seed)
        elif selected_type == 'SW':
            g = nx.connected_watts_strogatz_graph(n=cur_n, 
                                            k=self.g_params.get('k', random.choice([6,8,10,12,14,16])), 
                                            p=self.g_params.get('p', random.choice([0.05,0.15,0.25,0.35])),
                                            seed =seed)
        elif selected_type == 'BA':
            g = nx.barabasi_albert_graph(n=cur_n, 
                                        m=self.g_params.get('m', random.choice([3,4,5,6])), 
                                        seed=seed)
        elif selected_type == 'SBM':
            # Stochastic Block Model with 4 communities
            n_communities = 4
            sizes = [cur_n // n_communities] * n_communities
            # Adjust last community size to match cur_n exactly
            sizes[-1] += cur_n - sum(sizes)
            # Intra-community probability higher than inter-community
            p_in = 0.25
            p_out = 0.05
            probs = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]
            g = nx.stochastic_block_model(sizes, probs, seed=np.random.randint(1,1000))
        elif selected_type == 'ego':
            print("please implement one ego graph generation")
            sys.exit()
        else:
            raise ValueError(f"Unknown graph type: {selected_type}")
        
        # Ensure graph is connected (take largest component if needed)
        # This is important for ER, PL, and SBM graphs which can be disconnected
        if not nx.is_connected(g):
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc).copy()
            # Relabel nodes to be consecutive integers starting from 0
            g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
        
        return g
    
    def augment_graph(self, g):
        """
        Apply graph augmentation techniques to add noise/perturbations
        
        Args:
            g: networkx graph
        
        Returns:
            augmented graph
        """
        aug_config = self.g_params.get('augmentation', {})
        g_aug = g.copy()
        n_nodes = g_aug.number_of_nodes()
        n_edges = g_aug.number_of_edges()
        
        if n_edges == 0:
            return g_aug
        
        # 1. Random edge dropping
        drop_edge_prob = aug_config.get('drop_edge_prob', 0.05)
        if drop_edge_prob > 0:
            edges_to_drop = [e for e in list(g_aug.edges()) if random.random() < drop_edge_prob]
            g_aug.remove_edges_from(edges_to_drop)
        
        # 2. Random edge addition
        add_edge_prob = aug_config.get('add_edge_prob', 0.03)
        if add_edge_prob > 0 and n_nodes > 1:
            max_new_edges = max(1, int(n_edges * add_edge_prob))
            nodes = list(g_aug.nodes())
            attempts = 0
            max_attempts = max_new_edges * 10  # Avoid infinite loop
            added = 0
            while added < max_new_edges and attempts < max_attempts:
                u, v = random.sample(nodes, 2)
                if not g_aug.has_edge(u, v) and u != v:
                    g_aug.add_edge(u, v)
                    added += 1
                attempts += 1
        
        # 3. Random walk-based perturbation (optional, more aggressive)
        use_random_walk = aug_config.get('use_random_walk', False)
        if use_random_walk and n_nodes > 10:
            walk_length = aug_config.get('walk_length', min(20, n_nodes // 2))
            if len(list(g_aug.nodes())) > 0:
                start_node = random.choice(list(g_aug.nodes()))
                walk_nodes = self._random_walk(g_aug, start_node, walk_length)
                # Keep nodes in walk + their neighbors
                keep_nodes = set(walk_nodes)
                for node in walk_nodes:
                    keep_nodes.update(g_aug.neighbors(node))
                if len(keep_nodes) >= 5:  # Ensure minimum graph size
                    g_aug = g_aug.subgraph(keep_nodes).copy()
        
        # 4. Ensure graph is still connected (optional but recommended)
        ensure_connected = aug_config.get('ensure_connected', True)
        if ensure_connected and n_nodes > 1:
            if not nx.is_connected(g_aug):
                # Take largest connected component
                largest_cc = max(nx.connected_components(g_aug), key=len)
                g_aug = g_aug.subgraph(largest_cc).copy()
        
        # Relabel nodes to be consecutive integers starting from 0
        g_aug = nx.convert_node_labels_to_integers(g_aug, first_label=0, ordering='default')
        
        return g_aug
    
    def _random_walk(self, g, start_node, length):
        """
        Perform random walk from start_node
        
        Args:
            g: networkx graph
            start_node: starting node for walk
            length: number of steps
        
        Returns:
            list of nodes visited in the walk
        """
        walk = [start_node]
        current = start_node
        for _ in range(length - 1):
            neighbors = list(g.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)
        return walk

    def gen_new_graphs(self, num_min, num_max):
        print('Generating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        
        # Get augmentation probability (only augment a fraction of graphs)
        aug_prob = self.g_params.get('augmentation', {}).get('aug_probability', 0.5) if self.IsDisturbG else 0.0
        
        if self.g_type in ['ER','PL','SW','BA','mix']:
            for i in tqdm(range(1000), desc="Training graphs"):
                max_n = self.num_max
                min_n = self.num_min
                cur_n = np.random.randint(max_n - min_n + 1) + min_n
                g = self.gen_graph(cur_n)
                # Apply augmentation with probability aug_prob
                if self.IsDisturbG and random.random() < aug_prob:
                    g = self.augment_graph(g)
                self.InsertGraph(g, is_test=False)
                
        elif self.g_type in ['ego']:
            graphs = pickle.load(open(f"{self.g_params['train_dir']}/{self.g_params['target_graph']}_ego_train_{self.g_params['dataset_id']}.pkl", 'rb'))
            print(f"Loading training graphs from {self.g_params['train_dir']} (id: {self.g_params['dataset_id']})")
            self.g_params["dataset_id"] += 1
            for i in tqdm(range(1000), desc="Training graphs"):
                g = graphs[i]            
                # Apply augmentation with probability aug_prob
                if self.IsDisturbG and random.random() < aug_prob:
                    g = self.augment_graph(g)
                self.InsertGraph(g, is_test=False)


    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def PrepareValidData(self):
        print('Generating validation graphs...')
        sys.stdout.flush()
        cdef double result_degree = 0.0
        cdef double result_betweenness = 0.0
        
        # NOTE: Validation graphs are NOT augmented to ensure consistent evaluation
        if self.g_type in ['erdos_renyi','powerlaw','small-world','BA','mix']:
            for i in tqdm(range(n_valid), desc="Validation graphs"):
                max_n = self.num_max
                min_n = self.num_min
                cur_n = np.random.randint(max_n - min_n + 1) + min_n
                g = self.gen_graph(cur_n)
                g_degree = g.copy()
                g_betweenness = g.copy()
                val_degree, sol = self.HXA(g_degree, 'HDA')
                result_degree += val_degree
                val_betweenness, sol = self.HXA(g_betweenness, 'HBA')
                result_betweenness += val_betweenness
                self.InsertGraph(g, is_test=True)
                
        elif self.g_type in ['ego']:
            graphs = pickle.load(open(f"{self.g_params['valid_dir']}/{self.g_params['target_graph']}_ego_valid.pkl", 'rb'))
            print("Loading validation graphs from", self.g_params['valid_dir'])
            for i in tqdm(range(n_valid), desc="Validation graphs"):
                g = graphs[i]
                # g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
                g_degree = g.copy()
                g_betweenness = g.copy()
                val_degree, sol = self.HXA(g_degree, 'HDA')
                result_degree += val_degree
                val_betweenness, sol = self.HXA(g_betweenness, 'HBA')
                result_betweenness += val_betweenness
                self.InsertGraph(g, is_test=True)

        print('Validation HDA: %.6f, HBA: %.6f'%(result_degree / n_valid, result_betweenness / n_valid))


    def Run_simulator(self, int n_traj, double eps, TrainSet, int n_step):
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        while n < n_traj: 
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                        #print ('add experience transition!')
                    g_sample= TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= n_traj:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)
    #pass
    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        self.inputs['batch_graph_ids'] = prepareBatchGraph.batch_graph_ids
        # print("self.inputs = ",self.inputs)


    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        self.inputs['batch_graph_ids'] = prepareBatchGraph.batch_graph_ids
        return prepareBatchGraph.idx_map_list

    def Predict(self,g_list,covered,isSnapSnot):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, self.batch_size):
            bsize = self.batch_size
            if (i + self.batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            
            # Use sparse tensors directly - conversion handled in encoder/decoder
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.batch_graph_ids] = self.inputs['batch_graph_ids']

            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict = my_dict)
            else:
                result = self.session.run([self.q_on_all], feed_dict = my_dict)
            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,isSnapSnot=False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,isSnapSnot=True)
        return result
    #pass
    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        '''
        the sample contains:
        g_list: [BATCH_SIZE] (each element is a graph object)
        list_st: [BATCH_SIZE] (each element is a list of covered node indices, variable length)
        list_at: [BATCH_SIZE] (each element is an int, the action taken)
        list_rt: [BATCH_SIZE] (each element is a float, the reward)
        list_s_primes: [BATCH_SIZE] (each element is a list, the next state)
        list_term: [BATCH_SIZE] (each element is a bool, whether the episode ended)
        
        If using prioritized replay, you may also have:
            b_idx: batch indices in the replay buffer
            ISWeights: importance sampling weights, shape [BATCH_SIZE]
        
        '''
        sample = self.nStepReplayMem.Sampling(self.batch_size)
        ness = False
        cdef int i
        for i in range(self.batch_size):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
        
        # [BATCH_SIZE, 1], TD target for each sample
        list_target = np.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            q_rhs = 0
            gamma_n = self.gamma ** N_STEP  # Use GAMMA^n for n-step
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=gamma_n * list_pred[i]
                else:
                    q_rhs=gamma_n * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i] # TD target =  R_t + gamma^N_STEP * max(Q_pred)
            list_target[i] = q_rhs
        
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(
                tree_idx = sample.b_idx,
                ISWeights = sample.ISWeights,
                g_list=sample.g_list, 
                covered=sample.list_st, 
                actions=sample.list_at,
                list_target=list_target
                )
        else:
            return self.fit(
                g_list = sample.g_list, 
                covered = sample.list_st, 
                actions = sample.list_at,
                list_target = list_target
                )

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,self.batch_size):
            # batch_idxes is an array of indices for the current mini-batch, 
            # e.g., [0, 1, 2, ..., bsize-1] for each sub-batch within the full batch
            # For batch_size 64, the first batch_idxes would be [0, 1, ..., 63]
            bsize = self.batch_size
            if (i + self.batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            my_dict[self.target] = self.inputs['target']
            my_dict[self.batch_graph_ids] = self.inputs['batch_graph_ids']

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)


    def fit(self,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        
        for i in range(0,n_graphs,self.batch_size):
            bsize = self.batch_size
            if (i + self.batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            
            # Use sparse tensors directly - conversion handled in encoder/decoder
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.target] = self.inputs['target']
            my_dict[self.batch_graph_ids] = self.inputs['batch_graph_ids']

            result = self.session.run([self.loss,self.trainStep],feed_dict=my_dict)
            # print(result[0],bsize)
            loss += result[0].mean()*bsize
        return loss / len(g_list)


    def Train(self):
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
                N_start = N_end
                sys.stdout.flush()
                ########--------- Save --------########
                ckpt_file = self.ckpt_file_name(iter)
                self.SaveModel(ckpt_file)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            ########--------- Fit --------########    
            self.Fit()
        f_out.close()

    def Test(self,int gid):
        # Test when train
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        cdef int i
        sol = []
        while (not self.test_env.isTerminal()):
            # cost += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness

    def findModel(self):
        # find ckpt file in the model dir
        # return ckpt file name, not include dir e.g graphsage_iter_0.ckpt
        vc_list = []
        for line in open(self.VCFile):
            line=line.split(",")[1]
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        # best_model = os.path.join(cfd,'models/%s/nrange_%d_%d_iter_%d.ckpt' % (self.g_type, NUM_MIN, NUM_MAX, best_model_iter))
        best_model = self.ckpt_file_name(best_model_iter)
        print("Finding best model by validation score: %s"%best_model)
        return best_model
    
    def findLatestModel(self):
        # return ckpt file name
        vc_list = []
        for line in open(self.VCFile):
            line=line.split(",")[1]
            vc_list.append(float(line))
        last_iter = vc_list[-1]
        latest_model = self.ckpt_file_name(last_iter)
        return latest_model


    def Evaluate(self, test_graphs):
        # evaluate on a number of graphs
        sys.stdout.flush()

        g_num = len(test_graphs)
        cdef int n_test = g_num
        cdef int i
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = test_graphs[i]
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        self.ClearTestGraphs()
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std


    def EvaluateRealData(self, test_graph, sol_file=None, stepRatio=0.0025):  
        # evaluate on a graph, mostly used when eval real graph
        # save sol in sol_file
        sys.stdout.flush()

        cdef double solution_time = 0.0
        g = test_graph
        
        print ('testing')
        sys.stdout.flush()
        print ('number of nodes:%d'%(nx.number_of_nodes(g)))
        print ('number of edges:%d'%(nx.number_of_edges(g)))
        if stepRatio > 0:
            step = np.max([int(stepRatio*nx.number_of_nodes(g)),1]) #step size
        else:
            step = 1
        self.InsertGraph(g, is_test=True)
        t1 = time.time()
        solution = self.GetSolution(0,step)
        t2 = time.time()
        solution_time = (t2 - t1)

        if sol_file:
            with open(sol_file, 'w') as f_out:
                for i in range(len(solution)):
                    f_out.write('%d\n' % solution[i])
        
        self.ClearTestGraphs()
        return solution, solution_time


    def GetSolution(self, int gid, int step=1):
        # inner function, used inside the class
        # use TestSet to initialize testEnv
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        sum_sort_time = 0
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            start_time = time.time()
            batchSol = np.argsort(-list_pred[0])[:step]
            end_time = time.time()
            sum_sort_time += (end_time-start_time)
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        return sol

    def EvaluateSol(self, test_graph, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution and dataset, strategyID:0,count;2:rank;3:multipy
        sys.stdout.flush()
        g = test_graph
        g_inner = self.GenNetwork(g)
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            if reInsertStep > 0 and reInsertStep < 1:
                step = np.max([int(reInsertStep*nx.number_of_nodes(g)),1]) #step size
            else:
                step = reInsertStep
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, step)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution))
        Robustness = self.utils.getRobustness(g_inner, solution)
        MaxCCList = self.utils.MaxWccSzList
        return Robustness, MaxCCList

    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol


    def SaveModel(self,ckpt_file):
        model_path = os.path.join(self.save_model_dir,ckpt_file)
        self.saver.save(self.session, model_path)
        print(f'{model_path} has been saved success!\n')

    def LoadModel(self,ckpt_file):
        # ckpt_file: e.g. graphSage_iter_0.ckpt
        model_path = os.path.join(self.save_model_dir, ckpt_file)
        self.saver.restore(self.session, model_path)
        print(f'restore model from {model_path} successfully')

    def LoadModel_iter(self,iter):
        # iter: e.g. 0
        model_path = os.path.join(self.save_model_dir, self.ckpt_file_name(iter))
        self.saver.restore(self.session, model_path)
        print(f'restore model from {model_path} successfully')
    
    def ckpt_file_name(self,iter):
        # iter:int e.g. 100
        return '%s_iter_%d.ckpt' % (self.embeddingMethod, iter)

    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best


    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', ''
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        return Robustness, sol

    def resume_checkpoint_and_iter(self):
        # Path to the checkpoint file
        checkpoint_path = os.path.join(self.save_model_dir, f"{self.embeddingMethod}_{self.num_min}_{self.num_max}_checkpoint")
        if not os.path.exists(checkpoint_path):
            return None, 0
        with open(checkpoint_path, "r") as f:
            for line in f:
                if line.startswith("model_checkpoint_path:"):
                    ckpt = line.split(":")[1].strip().strip('"')
                    ckpt_name = os.path.basename(ckpt)
                    # Extract iteration number from filename, e.g., nrange_30_120_iter_116700.ckpt
                    import re
                    m = re.search(r"iter_(\d+)\.ckpt", ckpt_name)
                    if m:
                        last_iter = int(m.group(1))
                    else:
                        last_iter = 0
                    return ckpt, last_iter
        return None, 0

    def resume_nlines_and_runtime(self):
        if not os.path.exists(self.VCFile):
            return 0
        with open(self.VCFile, "r") as f:
            lines = f.readlines()
            if not lines:
                return 0
            last_line = lines[-1].strip()
            # The first value is the validation result, the second is the time, but we care about the line number
            runtime = float(last_line.split(",")[-1])
            return len(lines),runtime
