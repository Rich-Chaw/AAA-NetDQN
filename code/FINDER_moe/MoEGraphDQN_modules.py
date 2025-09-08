import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

import numpy as np
import networkx as nx
import random

class RouterNetwork:
    """Router network that decides which experts to activate based on graph characteristics"""
    
    def __init__(self, input_size=4, num_experts=4, top_k=2):
        self.input_size = input_size  # graph features: [num_nodes, m_param, density, avg_degree]
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router MLP
        self.router_mlp1 = tf1.Variable(
            tf1.truncated_normal([input_size, 64], stddev=0.01), tf.float32
        )
        self.router_bias1 = tf1.Variable(tf.zeros([64]), tf.float32)
        
        self.router_mlp2 = tf1.Variable(
            tf1.truncated_normal([64, num_experts], stddev=0.01), tf.float32
        )
        self.router_bias2 = tf1.Variable(tf.zeros([num_experts]), tf.float32)
    
    def route(self, graph_features):
        """
        Route input to experts
        Args:
            graph_features: [batch_size, input_size] - graph characteristics
        Returns:
            expert_weights: [batch_size, num_experts] - routing weights
            expert_mask: [batch_size, num_experts] - binary mask for top-k experts
        """
        # Router forward pass
        h = tf.nn.relu(tf.matmul(graph_features, self.router_mlp1) + self.router_bias1)
        expert_logits = tf.matmul(h, self.router_mlp2) + self.router_bias2
        
        # Get top-k experts
        expert_weights = tf.nn.softmax(expert_logits, axis=-1)
        
        # Create mask for top-k experts
        top_k_values, top_k_indices = tf.nn.top_k(expert_weights, k=self.top_k)
        expert_mask = tf.reduce_sum(
            tf.one_hot(top_k_indices, depth=self.num_experts, axis=-1), 
            axis=1
        )
        
        # Normalize weights for selected experts only
        masked_weights = expert_weights * expert_mask
        normalized_weights = masked_weights / (tf.reduce_sum(masked_weights, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_weights, expert_mask

class MoEGraphEncoder:
    """Mixture of Experts Graph Encoder with specialized experts for different graph types"""
    
    def __init__(self, embedding_size, feature_size, initialization_stddev, gnn_layers=3):
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.initialization_stddev = initialization_stddev
        self.gnn_layers = gnn_layers
        self.num_experts = 4
        
        # Router network
        self.router = RouterNetwork(input_size=4, num_experts=self.num_experts, top_k=2)
        
        # Expert 1: Specialized for sparse BA graphs (low m parameter)
        self.expert_sparse = self._create_expert('graphSage', 'sparse')
        
        # Expert 2: Specialized for dense BA graphs (high m parameter)  
        self.expert_dense = self._create_expert('GIN', 'dense')
        
        # Expert 3: Specialized for small graphs (n < 50)
        self.expert_small = self._create_expert('S2V', 'small')
        
        # Expert 4: Specialized for large graphs (n > 100)
        self.expert_large = self._create_expert('graphSage', 'large')
        
        self.experts = [self.expert_sparse, self.expert_dense, self.expert_small, self.expert_large]
    
    def _create_expert(self, embedding_method, expert_name):
        """Create a specialized expert"""
        return GraphEncoder(
            embeddingMethod=embedding_method,
            feature_size=self.feature_size,
            embedding_size=self.embedding_size,
            initialization_stddev=self.initialization_stddev,
            gnn_layers=self.gnn_layers,
            expert_name=expert_name
        )
    
    def extract_graph_features(self, n2nsum_param, subgsum_param, batch_graph_ids):
        """Extract graph characteristics for routing"""
        num_graphs = tf.shape(subgsum_param)[0]
        num_nodes = tf.shape(n2nsum_param)[0]
        
        # Calculate graph features
        # 1. Number of nodes per graph
        nodes_per_graph = tf.math.unsorted_segment_sum(
            tf.ones([num_nodes]), batch_graph_ids, num_graphs
        )
        
        # 2. Graph density (approximate)
        edge_count = tf.reduce_sum(tf.cast(n2nsum_param, tf.float32))
        density = edge_count / (tf.cast(num_nodes, tf.float32) * tf.cast(num_nodes - 1, tf.float32))
        
        # 3. Average degree
        avg_degree = 2.0 * edge_count / tf.cast(num_nodes, tf.float32)
        
        # 4. M parameter (estimated from density)
        estimated_m = tf.cast(avg_degree / 2.0, tf.float32)
        
        # Normalize features
        normalized_nodes = tf.cast(nodes_per_graph, tf.float32) / 200.0  # normalize by max expected nodes
        normalized_density = tf.clip_by_value(density, 0.0, 1.0)
        normalized_avg_degree = tf.clip_by_value(avg_degree / 20.0, 0.0, 1.0)  # normalize by max expected degree
        normalized_m = tf.clip_by_value(estimated_m / 10.0, 0.0, 1.0)  # normalize by max expected m
        
        graph_features = tf.stack([
            normalized_nodes,
            normalized_m, 
            normalized_density,
            normalized_avg_degree
        ], axis=1)
        
        return graph_features
    
    def encode(self, node_input, y_node_input, n2nsum_param, subgsum_param, batch_graph_ids=None):
        """
        Encode graphs using MoE architecture
        """
        # Extract graph features for routing
        graph_features = self.extract_graph_features(n2nsum_param, subgsum_param, batch_graph_ids)
        
        # Route to experts
        expert_weights, expert_mask = self.router.route(graph_features)
        
        # Get expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            cur_message_layer, y_cur_message_layer = expert.encode(
                node_input, y_node_input, n2nsum_param, subgsum_param, batch_graph_ids
            )
            expert_outputs.append((cur_message_layer, y_cur_message_layer))
        
        # Combine expert outputs using routing weights
        batch_size = tf.shape(subgsum_param)[0]
        num_nodes = tf.shape(n2nsum_param)[0]
        
        # Initialize combined outputs
        combined_node_embeddings = tf.zeros([num_nodes, self.embedding_size])
        combined_graph_embeddings = tf.zeros([batch_size, self.embedding_size])
        
        # Weighted combination
        for i, (node_emb, graph_emb) in enumerate(expert_outputs):
            # Expand weights for node-level combination
            node_weights = tf.gather(expert_weights[:, i], batch_graph_ids)
            node_weights = tf.expand_dims(node_weights, axis=1)
            
            # Combine node embeddings
            weighted_node_emb = node_emb * node_weights
            combined_node_embeddings += weighted_node_emb
            
            # Combine graph embeddings
            graph_weights = tf.expand_dims(expert_weights[:, i], axis=1)
            weighted_graph_emb = graph_emb * graph_weights
            combined_graph_embeddings += weighted_graph_emb
        
        return combined_node_embeddings, combined_graph_embeddings, expert_weights

class MoEMLPDecoder:
    """MLP Decoder for MoE architecture"""
    
    def __init__(self, embedding_size, reg_hidden, aux_dim, initialization_stddev):
        self.embedding_size = embedding_size
        self.reg_hidden = reg_hidden
        self.aux_dim = aux_dim
        self.initialization_stddev = initialization_stddev
        
        # Q-value prediction layers
        self.q_w1 = tf1.Variable(
            tf1.truncated_normal([embedding_size * 2 + aux_dim, reg_hidden], stddev=initialization_stddev), 
            tf.float32
        )
        self.q_b1 = tf1.Variable(tf.zeros([reg_hidden]), tf.float32)
        
        self.q_w2 = tf1.Variable(
            tf1.truncated_normal([reg_hidden, 1], stddev=initialization_stddev), 
            tf.float32
        )
        self.q_b2 = tf1.Variable(tf.zeros([1]), tf.float32)
        
        # Q-value prediction for all nodes
        self.q_all_w1 = tf1.Variable(
            tf1.truncated_normal([embedding_size * 2 + aux_dim, reg_hidden], stddev=initialization_stddev), 
            tf.float32
        )
        self.q_all_b1 = tf1.Variable(tf.zeros([reg_hidden]), tf.float32)
        
        self.q_all_w2 = tf1.Variable(
            tf1.truncated_normal([reg_hidden, 1], stddev=initialization_stddev), 
            tf.float32
        )
        self.q_all_b2 = tf1.Variable(tf.zeros([1]), tf.float32)
    
    def decode_q(self, node_embeddings, action_select, graph_embeddings, aux_input):
        """
        Decode Q-values for selected actions
        """
        # Get embeddings for selected actions
        action_embeddings = tf1.sparse_tensor_dense_matmul(action_select, node_embeddings)
        
        # Concatenate action embeddings, graph embeddings, and auxiliary features
        combined_input = tf.concat([action_embeddings, graph_embeddings, aux_input], axis=1)
        
        # MLP forward pass
        h = tf.nn.relu(tf.matmul(combined_input, self.q_w1) + self.q_b1)
        q_values = tf.matmul(h, self.q_w2) + self.q_b2
        
        return q_values
    
    def decode_q_all(self, node_embeddings, graph_embeddings, aux_input, rep_global):
        """
        Decode Q-values for all nodes
        """
        # Expand graph embeddings to match node embeddings
        batch_size = tf.shape(graph_embeddings)[0]
        num_nodes = tf.shape(node_embeddings)[0]
        
        # Use rep_global to get graph embeddings for each node
        graph_emb_per_node = tf1.sparse_tensor_dense_matmul(rep_global, graph_embeddings)
        
        # Concatenate node embeddings, graph embeddings, and auxiliary features
        combined_input = tf.concat([node_embeddings, graph_emb_per_node, aux_input], axis=1)
        
        # MLP forward pass
        h = tf.nn.relu(tf.matmul(combined_input, self.q_all_w1) + self.q_all_b1)
        q_values_all = tf.matmul(h, self.q_all_w2) + self.q_all_b2
        
        return q_values_all
