#!/usr/bin/env python3
"""
FINDER_ND Fine-tuning Script
Fine-tune pre-trained GraphDQN model on real graphs with two strategies:
1. Freezing mode: Freeze encoder, train only decoder
2. Full fine-tuning mode: Train both encoder and decoder end-to-end
"""

import os
import sys
import numpy as np
import networkx as nx
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from graph_dqn_modules import GraphEncoder, MLPDecoder
import PrepareBatchGraph
# from GraphDQN import GraphDQN

class FineTuneGraphDQN:
    def __init__(self, 
                 pretrained_ckpt_path='./models/BA_sota/graphSage_nrange_30_50_iter_78000.ckpt',
                 save_model_dir = './models/finetune',
                 real_graph_path='dataset/real/Digg.txt',
                 embedding_method='GIN',
                 embedding_size=64,
                 feature_size=2,
                 reg_hidden=64,
                 aux_dim=4,
                 initialization_stddev=0.01,
                 gnn_layers=3,
                 learning_rate=0.001,
                 batch_size=32,
                 num_epochs=100,
                 fine_tune_mode='full'):  # 'freeze' or 'full'
        
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.save_model_dir = save_model_dir
        self.real_graph_path = real_graph_path
        self.embedding_method = embedding_method
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.reg_hidden = reg_hidden
        self.aux_dim = aux_dim
        self.initialization_stddev = initialization_stddev
        self.gnn_layers = gnn_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.fine_tune_mode = fine_tune_mode
        
        # Initialize components
        self.encoder = None
        self.decoder = None
        self.saver = None
        self.session = None
        
        # Training placeholders
        self.node_input = None
        self.y_node_input = None
        self.n2nsum_param = None
        self.subgsum_param = None
        self.aux_input = None
        self.action_select = None
        self.rep_global = None
        self.batch_graph_ids = None
        self.labels = None
        
        # Training ops
        self.loss_op = None
        self.train_op = None
        self.pred_op = None
        
        print(f"Initializing FineTuneGraphDQN in {fine_tune_mode} mode")
        
    def load_real_graph(self):
        """Load real graph from file"""
        print(f"Loading real graph from {self.real_graph_path}")
        
        # Parse graph file (assuming edge list format)
        edges = []
        with open(self.real_graph_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        edges.append((u, v))
        
        # Create NetworkX graph
        graph = nx.Graph(edges)
        print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Convert to FINDER_ND format
        class RealGraph:
            def __init__(self, nx_graph):
                self.nx_graph = nx_graph
                self.num_nodes = nx_graph.number_of_nodes()
                self.num_edges = nx_graph.number_of_edges()
                self.edge_list = list(nx_graph.edges())
                self.node_list = list(nx_graph.nodes())
                
                # Create adjacency lists
                self.adj_list = {}
                for node in nx_graph.nodes():
                    self.adj_list[node] = list(nx_graph.neighbors(node))
        
        return RealGraph(graph)
    
    def create_graph_batches(self, real_graph, num_batches=10):
        """Create batches of subgraphs from the real graph"""
        print("Creating graph batches for training...")
        
        batches = []
        nodes = list(real_graph.nx_graph.nodes())
        
        for i in range(num_batches):
            # Sample a random subgraph of size 30-50 nodes
            subgraph_size = np.random.randint(30, 51)
            if subgraph_size > len(nodes):
                subgraph_size = len(nodes)
            
            # Randomly sample nodes
            sampled_nodes = np.random.choice(nodes, subgraph_size, replace=False)
            
            # Create subgraph
            subgraph = real_graph.nx_graph.subgraph(sampled_nodes)
            
            # Convert to graph.cpp format
            class SubGraph:
                def __init__(self, nx_subgraph):
                    self.nx_graph = nx_subgraph
                    self.num_nodes = nx_subgraph.number_of_nodes()
                    self.num_edges = nx_subgraph.number_of_edges()
                    self.edge_list = list(nx_subgraph.edges())
                    self.node_list = list(nx_subgraph.nodes())
                    
                    # Create adjacency lists
                    self.adj_list = {}
                    for node in nx_subgraph.nodes():
                        self.adj_list[node] = list(nx_subgraph.neighbors(node))
            
            batches.append(SubGraph(subgraph))
        
        print(f"Created {len(batches)} graph batches")
        return batches
    
    def build_model(self):
        """Build the fine-tuning model"""
        print("Building fine-tuning model...")
        
        # Create TensorFlow session
        self.session = tf1.Session()
        
        # Initialize components
        self.encoder = GraphEncoder(
            embeddingMethod=self.embedding_method,
            feature_size=self.feature_size,
            embedding_size=self.embedding_size,
            initialization_stddev=self.initialization_stddev,
            gnn_layers=self.gnn_layers
        )
        
        self.decoder = MLPDecoder(
            embedding_size=self.embedding_size,
            reg_hidden=self.reg_hidden,
            aux_dim=self.aux_dim,
            initialization_stddev=self.initialization_stddev
        )
        
        # Create placeholders
        self.node_input = tf1.placeholder(tf.float32, [None, self.feature_size], name='node_input')
        self.y_node_input = tf1.placeholder(tf.float32, [None, self.feature_size], name='y_node_input')
        
        self.n2nsum_param = tf1.sparse_placeholder(tf.float32, [None, None], name='n2nsum_param')
        self.subgsum_param = tf1.sparse_placeholder(tf.float32, [None, None], name='subgsum_param')
        self.action_select = tf1.sparse_placeholder(tf.float32, [None, None], name='action_select')
        self.rep_global = tf1.sparse_placeholder(tf.float32, [None, None], name='rep_global')
        self.aux_input = tf1.placeholder(tf.float32, [None, self.aux_dim], name='aux_input')
        self.batch_graph_ids = tf1.placeholder(tf.int32, [None], name='batch_graph_ids')
        
        self.labels = tf1.placeholder(tf.float32, [None, 1], name='labels')
        
        # Build forward pass
        with tf1.variable_scope('encoder'):
            node_embeddings, graph_embeddings = self.encoder.encode(
                self.node_input,
                self.y_node_input,
                self.n2nsum_param,
                self.subgsum_param,
                self.batch_graph_ids
            )
        
        with tf1.variable_scope('decoder'):
            predictions = self.decoder.decode_q(
                node_embeddings,
                self.action_select,
                graph_embeddings,
                self.aux_input
            )
        
        # Loss and training ops
        self.loss_op = tf1.reduce_mean(tf1.square(predictions - self.labels))
        self.pred_op = predictions
        
        # Set up training based on fine-tuning mode
        if self.fine_tune_mode == 'freeze':
            all_vars = tf1.trainable_variables()
            # all_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)  # same as above
              
            # Only train decoder parameters
            # decoder_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder') # DO NOT use, cause Variable do not have name
            decoder_vars = [self.decoder.h1_weight, self.decoder.last_w, self.decoder.cross_product]
            self.train_op = tf1.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss_op, var_list=decoder_vars
            )
            
            print("Freezing encoder parameters, training only decoder")
            
        else:  # full fine-tuning
            # Train all parameters
            self.train_op = tf1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)
            print("Training both encoder and decoder end-to-end")
        
        # Initialize variables
        self.session.run(tf1.global_variables_initializer())
        
        # Create saver
        self.saver = tf1.train.Saver()
        
        print("Model built successfully")
    
    def load_pretrained_weights(self):
        """Load pre-trained weights from checkpoint"""
        print(f"Loading pre-trained weights from {self.pretrained_ckpt_path}")
        
        try:
            # # Find the latest checkpoint
            # ckpt_files = []
            # for file in os.listdir(self.pretrained_ckpt_path):
            #     if file.endswith('.ckpt'):
            #         ckpt_files.append(file)
            
            # if not ckpt_files:
            #     print("No checkpoint files found!")
            #     return False
            
            # # Load the latest checkpoint
            # latest_ckpt = os.path.join(self.pretrained_ckpt_path, ckpt_files[-1])
            latest_ckpt = self.pretrained_ckpt_path
            self.saver.restore(self.session, latest_ckpt)
            print(f"Loaded pre-trained weights from {latest_ckpt}")
            return True
            
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            return False
    
    def prepare_batch_data(self, graph_batch, covered_nodes=None):
        """Prepare batch data for training"""
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID=0)

        if covered_nodes is None:
            covered_nodes = []
        
        # Setup batch graph
        prepareBatchGraph.SetupPredAll(
            idxes=[0],  # Single graph
            g_list=[graph_batch],
            covered=[covered_nodes]
        )
        
        # Extract data
        node_input = np.ones((prepareBatchGraph.graph.num_nodes, self.feature_size))
        y_node_input = np.ones((1, self.feature_size))
        

        n2nsum_param = prepareBatchGraph.n2nsum_param
        subgsum_param = prepareBatchGraph.subgsum_param

        # Create auxiliary features
        aux_input = prepareBatchGraph.aux_feat[0]
        # Create global representation matrix
        rep_global = prepareBatchGraph.rep_global

        # Create action selection matrix (for demonstration, select first available node)
        action_select = np.zeros((1, prepareBatchGraph.graph.num_nodes))
        if prepareBatchGraph.avail_act_cnt[0] > 0:
            action_select[0, 0] = 1.0  # Select first available node
        
        # Create batch graph IDs
        batch_graph_ids = prepareBatchGraph.batch_graph_ids
        
        return {
            'node_input': node_input,
            'y_node_input': y_node_input,
            'n2nsum_param': n2nsum_param,
            'subgsum_param': subgsum_param,
            'aux_input': aux_input,
            'action_select': action_select,
            'rep_global': rep_global,
            'batch_graph_ids': batch_graph_ids
        }
    
    def generate_training_data(self, graph_batches):
        """Generate training data with synthetic labels"""
        print("Generating training data...")
        
        training_data = []
        
        for graph_batch in graph_batches:
            # Create multiple training examples per graph
            for _ in range(5):  # 5 examples per graph
                # Randomly cover some nodes
                num_covered = np.random.randint(0, min(10, graph_batch.num_nodes))
                covered_nodes = np.random.choice(
                    list(range(graph_batch.num_nodes)), 
                    num_covered, 
                    replace=False
                ).tolist()
                
                # Prepare batch data
                batch_data = self.prepare_batch_data(graph_batch, covered_nodes)
                
                # Generate synthetic label (for demonstration)
                # In real scenario, this would be the actual reward/objective
                label = np.random.uniform(0, 1, (1, 1))
                
                batch_data['label'] = label
                training_data.append(batch_data)
        
        print(f"Generated {len(training_data)} training examples")
        return training_data
    
    def train(self):
        """Main training loop"""
        print(f"Starting fine-tuning in {self.fine_tune_mode} mode...")
        
        # Load real graph
        real_graph = self.load_real_graph()
        
        # Create graph batches
        graph_batches = self.create_graph_batches(real_graph, num_batches=50)
        
        # Build model
        self.build_model()
        
        # Load pre-trained weights
        if not self.load_pretrained_weights():
            print("Warning: Could not load pre-trained weights. Starting from scratch.")
        
        # Generate training data
        training_data = self.generate_training_data(graph_batches)
        
        # Training loop
        print(f"Training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            np.random.shuffle(training_data)
            
            # Process in batches
            for i in range(0, len(training_data), self.batch_size):
                batch_data = training_data[i:i+self.batch_size]
                
                # Training step
                feed_dict = {
                    self.node_input: batch_data['node_input'],
                    self.y_node_input: batch_data['y_node_input'],
                    self.n2nsum_param: batch_data['n2nsum_param'],
                    self.subgsum_param: batch_data['subgsum_param'],
                    self.aux_input: batch_data['aux_input'],
                    self.action_select: batch_data['action_select'],
                    self.rep_global: batch_data['rep_global'],
                    self.batch_graph_ids: batch_data['batch_graph_ids'],
                    self.labels: batch_data['labels']
                }
                
                loss, _ = self.session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
                epoch_loss += loss
                num_batches += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss:.6f}")
        
        print("Fine-tuning completed!")
        
        # Save fine-tuned model
        save_path = os.path.join(self.save_model_dir, f'finetuned_{self.fine_tune_mode}')
        self.saver.save(self.session, save_path)
        print(f"Fine-tuned model saved to {save_path}")
    
    def evaluate(self, test_graphs):
        """Evaluate the fine-tuned model"""
        print("Evaluating fine-tuned model...")
        
        total_loss = 0.0
        num_examples = 0
        
        for graph in test_graphs:
            # Generate test examples
            for _ in range(3):  # 3 test examples per graph
                covered_nodes = np.random.choice(
                    list(range(graph.num_nodes)), 
                    np.random.randint(0, min(5, graph.num_nodes)), 
                    replace=False
                ).tolist()
                
                batch_data = self.prepare_batch_data(graph, covered_nodes)
                
                # Generate synthetic test label
                test_label = np.random.uniform(0, 1, (1, 1))
                
                # Forward pass
                feed_dict = {
                    self.node_input: batch_data['node_input'],
                    self.y_node_input: batch_data['y_node_input'],
                    self.n2nsum_param: batch_data['n2nsum_param'],
                    self.subgsum_param: batch_data['subgsum_param'],
                    self.aux_input: batch_data['aux_input'],
                    self.action_select: batch_data['action_select'],
                    self.rep_global: batch_data['rep_global'],
                    self.batch_graph_ids: batch_data['batch_graph_ids'],
                    self.labels: test_label
                }
                
                loss, pred = self.session.run([self.loss_op, self.pred_op], feed_dict=feed_dict)
                total_loss += loss
                num_examples += 1
        
        avg_loss = total_loss / num_examples
        print(f"Evaluation - Average Loss: {avg_loss:.6f}")
        return avg_loss

def main():
    """Main function to run fine-tuning"""
    print("===test fine-tuning Script on Digg dataset===")
    
    # Configuration
    config = {
        'pretrained_ckpt_path': '../models/BA_sota/graphSage_nrange_30_50_iter_78000.ckpt',
        'real_graph_path': '../../dataset/real/Digg.txt',
        'embedding_method': 'GIN',
        'embedding_size': 64,
        'feature_size': 2,
        'reg_hidden': 64,
        'aux_dim': 4,
        'initialization_stddev': 0.01,
        'gnn_layers': 3,
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_epochs': 50
    }
    
    # Run both fine-tuning modes
    for mode in ['freeze', 'full']:
        print(f"\n{'='*50}")
        print(f"Running {mode.upper()} fine-tuning mode")
        print(f"{'='*50}")
        
        config['fine_tune_mode'] = mode
        
        try:
            # Initialize fine-tuning
            finetuner = FineTuneGraphDQN(**config)
            # Train
            finetuner.train()
            
            # Evaluate (create some test graphs)
            real_graph = finetuner.load_real_graph()
            test_graphs = finetuner.create_graph_batches(real_graph, num_batches=10)
            finetuner.evaluate(test_graphs)
            
        except Exception as e:
            print(f"Error in {mode} fine-tuning: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()