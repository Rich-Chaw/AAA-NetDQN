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

class MoEGraphDQN:

    def __init__(self,
        **model_config
    ):
        # Extract parameters from model_config
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = model_config["g_type"] #BA(barabasi_albert),ER(),PL(powerlaw), SW(small-world), ego
        self.g_params = model_config["g_params"]
        self.num_min = int(self.g_params['nrange'].split('_')[0])
        self.num_max = int(self.g_params['nrange'].split('_')[1])
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()
        
        # MoE specific parameters
        self.moe_config = model_config.get("moe_config", {
            'num_experts': 4,
            'top_k': 2,
            'router_dropout': 0.1,
            'load_balance_loss_weight': 0.01,
            'from_pretrained': False,
            'pretrained_model_dir': "./FINDER/models"
        })
        self.top_k = self.moe_config['top_k']
        self.from_pretrained = self.moe_config['from_pretrained']
        self.pretrained_model_dir = self.moe_config['pretrained_model_dir']
        # num_experts will be determined after encoder creation
        self.num_experts = self.moe_config.get('num_experts', 4)  # Default fallback

        ############----------------------------- paths ------------------- ###################################
        # save_model_dir: directory to save the models
        save_model_dir = model_config["save_model_dir"]
        if self.g_type == 'ego' or self.g_type == 'mix':
            self.save_model_dir = f"{save_model_dir}/MoE_{self.g_type}_nrange_{self.g_params['nrange']}"
        else: 
            self.save_model_dir = f"{save_model_dir}/MoE_{self.g_type}_nrange_{self.g_params['nrange']}_m_{self.g_params['m']}"
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
            self.env_list.append(mvc_env.py_MvcEnv(self.num_max))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(self.num_max)
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
        self.target = tf1.placeholder(tf.float32, [BATCH_SIZE,1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf1.placeholder(tf.float32, name="aux_input")
        # [N] batch graph ids for routing
        self.batch_graph_ids = tf1.placeholder(tf.int32, name="batch_graph_ids")

        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf1.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')

        # Build the MoE network
        self.encoder, self.decoder, self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildMoENet()
        # init Target Q Network
        self.encoderT, self.decoderT,self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT = self.BuildMoENet(is_target = True)
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
        
        # Print GPU information (cleaner version)
        gpus = tf1.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {len(gpus)} device(s) available, Using device: {gpus[0].name}")
            print(f"tf.is_gpu_available() : {tf.test.is_gpu_available()}\n")
        else:
            print("[WARNING] No GPU detected - training will use CPU")

    def BuildMoENet(self,is_target=False):
        """Build the MoE network architecture - CREATES SEPARATE INSTANCES"""
        # N: number of nodes (of all graphs in a batch)
        nodes_size = tf.shape(self.n2nsum_param)[0]
        # B: batch_size (number of graphs in a batch)
        y_nodes_size = tf.shape(self.subgsum_param)[0]

        feature_size = 2 # 2 features for node and graph

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
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.TestSet.InsertGraph(t,self.GenNetwork(g))
            self.ngraph_test += 1
        else:
            t = self.ngraph_train
            self.TrainSet.InsertGraph(t,self.GenNetwork(g))
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

    # def PlayGame(self, int n_traj, double eps):
    #     """Play games to collect experience (same as original)"""
    #     cdef int i, j
    #     for i in range(n_traj):
    #         self.env_list[0].s0(self.TrainSet.Get(i % self.ngraph_train))
    #         self.g_list[0] = self.env_list[0].graph
    #         while (not self.env_list[0].isTerminal()):
    #             list_pred = self.PredictWithCurrentQNet(self.g_list, [self.env_list[0].action_list])
    #             if random.random() < eps:
    #                 action = random.choice(self.env_list[0].action_list)
    #             else:
    #                 action = self.argMax(list_pred[0])
    #             self.env_list[0].step(action)

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
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            
            # Get expert utilization (initialize with uniform distribution)
            # batch_size = len(g_list)
            expert_util = np.ones((bsize, self.num_experts)) / self.num_experts

            # Use sparse tensors directly - conversion handled in encoder/decoder
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.batch_graph_ids] = self.inputs['batch_graph_ids']
            # add expert_utilization
            # my_dict[self.expert_utilization] = expert_util

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
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result

    # def PredictWithCurrentQNet(self, g_list, action_list):
    #     """Predict Q-values with current network (adapted for MoE)"""
    #     # # Prepare batch data
    #     # batch_data = PrepareBatchGraph.py_PrepareBatchGraph(g_list, action_list)
        
    #     # # Get expert utilization (initialize with uniform distribution)
    #     # batch_size = len(g_list)
    #     # expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
        
    #     # # Run prediction
    #     # q_values = self.session.run(self.q_on_all, feed_dict={
    #     #     self.n2nsum_param: batch_data['n2nsum_param'],
    #     #     self.subgsum_param: batch_data['subgsum_param'],
    #     #     self.rep_global: batch_data['rep_global'],
    #     #     self.aux_input: batch_data['aux_input'],
    #     #     self.batch_graph_ids: batch_data['batch_graph_ids'],
    #     #     self.expert_utilization: expert_util
    #     # })
        
    #     # return q_values


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
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        cdef int i
        for i in range(BATCH_SIZE):
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
        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i] # TD target =  R_t + GAMMA * max(Q_pred)
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
        for i in range(0,n_graphs,BATCH_SIZE):
            # batch_idxes is an array of indices for the current mini-batch, 
            # e.g., [0, 1, 2, ..., bsize-1] for each sub-batch within the full batch
            # For batch_size 64, the first batch_idxes would be [0, 1, ..., 63]
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
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

            # Get expert utilization for load balancing
            # batch_size = len(batch_data['g_list'])
            # expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
            # my_dict[self.expert_utilization] = expert_util

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)


    
    def fit(self,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
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

            # # Get expert utilization for load balancing
            # batch_size = len(batch_data['g_list'])
            # expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
            # my_dict[self.expert_utilization] = expert_util

            result = self.session.run([self.loss,self.trainStep],feed_dict=my_dict)
            # print(result[0],bsize)
            loss += result[0].mean()*bsize
        return loss / len(g_list)

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

    # def Fit(self):
    #     """Fit the model (adapted for MoE)"""
    #     if self.nStepReplayMem.size() < BATCH_SIZE:
    #         return
        
    #     # Sample batch
    #     if self.IsPrioritizedSampling:
    #         batch_data, batch_ISWeights = self.nStepReplayMem.sample(BATCH_SIZE)
    #         self.ISWeights = batch_ISWeights
    #     else:
    #         batch_data = self.nStepReplayMem.sample(BATCH_SIZE)
        
    #     # Get expert utilization for load balancing
    #     batch_size = len(batch_data['g_list'])
    #     expert_util = np.ones((batch_size, self.num_experts)) / self.num_experts
        
    #     # Train
    #     self.session.run(self.trainStep, feed_dict={
    #         self.n2nsum_param: batch_data['n2nsum_param'],
    #         self.subgsum_param: batch_data['subgsum_param'],
    #         self.action_select: batch_data['action_select'],
    #         self.rep_global: batch_data['rep_global'],
    #         self.target: batch_data['target'],
    #         self.aux_input: batch_data['aux_input'],
    #         self.batch_graph_ids: batch_data['batch_graph_ids'],
    #         self.expert_utilization: expert_util
    #     })

    def TakeSnapShot(self):
        """Take snapshot of target network (same as original)"""
        self.session.run(self.UpdateTargetQNetwork)

    def ckpt_file_name(self,iter):
        # iter:int e.g. 100
        if self.from_pretrained:
            return 'MoE_pretrained_iter_%d.ckpt' % iter
        else:
            return 'MoE_iter_%d.ckpt' % iter

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

    def SaveModel(self, filename):
        """Save model checkpoint"""
        save_path = os.path.join(self.save_model_dir, filename)
        self.saver.save(self.session, save_path)
        print(f"Model saved to {save_path}")

    def LoadModel(self, filename):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_model_dir, filename)
        self.saver.restore(self.session, load_path)
        print(f"Model loaded from {load_path}")

    def LoadModel_iter(self,iter):
        # iter: e.g. 0
        model_path = os.path.join(self.save_model_dir, self.ckpt_file_name(iter))
        self.saver.restore(self.session, model_path)
        print(f'restore model from {model_path} successfully')

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
        # last_ckpt, last_iter = self.resume_checkpoint_and_iter()
        # if last_ckpt != None:
        #     print(f"Resuming from checkpoint: {last_ckpt} at iter {last_iter}")
        #     self.LoadModel(last_ckpt)
        #     start_iter = last_iter + 1
        #     _,runtime = self.resume_nlines_and_runtime()
        #     # Open CSV in append mode
        #     f_out = open(self.VCFile, 'a')
        # else:
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
                ckpt_file = self.ckpt_file_name(iter)
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
        if self.from_pretrained:
            best_model = 'MoE_pretrained_iter_%d.ckpt' % best_model_iter
        else:
            best_model = 'MoE_iter_%d.ckpt' % best_model_iter
        print("Finding best model by validation score: %s"%best_model)
        return best_model

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


    def EvaluateRealData(self, test_graph, result_file=None, stepRatio=0.0025):  
        # evaluate on a graph, mostly used when eval real graph
        # save sol in result_file
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

        if result_file:
            with open(result_file, 'w') as f_out:
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
