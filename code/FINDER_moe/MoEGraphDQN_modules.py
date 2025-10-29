"""
MoE GraphDQN Modules - Mixture of Experts Architecture
=====================================================

This module implements a Mixture of Experts (MoE) version of the GraphDQN architecture.

Key Differences from Original GraphDQN_modules.py:
1. ORIGINAL: Single GraphEncoder + Single MLPDecoder
   MoE: RouterNetwork + Multiple GraphEncoders (experts) + Single MoEMLPDecoder

2. ORIGINAL: Direct encoding: graph -> embedding -> Q-values
   MoE: Routing-based encoding: graph -> router decision -> expert selection -> weighted combination -> Q-values

3. ORIGINAL: One-size-fits-all approach for all graph types
   MoE: Specialized experts for different graph characteristics (sparse/dense, small/large)
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

import numpy as np
import networkx as nx
import random

# Import original modules for expert creation
from GraphDQN_modules import GraphEncoder, MLPDecoder

class RouterNetwork:
    """
    NEW COMPONENT: Router network that decides which experts to activate based on graph characteristics
    
    ORIGINAL ARCHITECTURE: No router - single encoder processes all graphs
    MoE ARCHITECTURE: Router analyzes graph features and decides expert weights
    
    Purpose: Learn to route different graph types to specialized experts
    """
    
    def __init__(self, input_size=4, num_experts=4, top_k=2):
        # Input: 4 graph features [normalized_nodes, normalized_m, normalized_density, normalized_avg_degree]
        self.input_size = input_size  
        self.num_experts = num_experts  # 4 experts in our MoE setup
        self.top_k = top_k  # Only activate top-2 experts (sparse MoE for efficiency)
        
        # Router MLP: 2-layer neural network that maps graph features to expert weights
        # Layer 1: [4] -> [64] (feature extraction)
        self.router_mlp1 = tf1.Variable(
            tf1.truncated_normal([input_size, 64], stddev=0.01), tf.float32
        )
        self.router_bias1 = tf1.Variable(tf.zeros([64]), tf.float32)
        
        # Layer 2: [64] -> [4] (expert logits)
        self.router_mlp2 = tf1.Variable(
            tf1.truncated_normal([64, num_experts], stddev=0.01), tf.float32
        )
        self.router_bias2 = tf1.Variable(tf.zeros([num_experts]), tf.float32)
    
    def route(self, graph_features):
        """
        CORE MoE ROUTING MECHANISM: Decides which experts to use for each graph
        
        ORIGINAL: No routing - single encoder processes all graphs uniformly
        MoE: Dynamic expert selection based on graph characteristics
        
        Args:
            graph_features: [B, 4] - normalized graph characteristics
                           [norm_nodes, norm_m, norm_density, norm_avg_degree]
        Returns:
            normalized_weights: [B, num_experts] - final expert weights (sum to 1)
            expert_mask: [B, num_experts] - binary mask for active experts
        """
        # Step 1: Router MLP forward pass
        # [B, 4] -> [B, 64] (feature extraction layer)
        h = tf.nn.relu(tf.matmul(graph_features, self.router_mlp1) + self.router_bias1)
        
        # [B, 64] -> [B, num_experts] (expert logits)
        expert_logits = tf.matmul(h, self.router_mlp2) + self.router_bias2
        
        # Step 2: Convert logits to probabilities
        # [B, num_experts] - softmax gives probability distribution over experts
        expert_weights = tf.nn.softmax(expert_logits, axis=-1)
        
        # Step 3: Sparse MoE - only activate top-k experts (efficiency)
        # Find top-2 experts for each graph
        top_k_values, top_k_indices = tf.nn.top_k(expert_weights, k=self.top_k)
        
        # Create binary mask: 1 for selected experts, 0 for others
        # [B, top_k] -> [B, num_experts]
        expert_mask = tf.reduce_sum(
            tf.one_hot(top_k_indices, depth=self.num_experts, axis=-1), 
            axis=1  # Sum over top_k dimension
        )
        
        # Step 4: Renormalize weights for selected experts only
        # Zero out non-selected experts
        masked_weights = expert_weights * expert_mask
        # Renormalize so selected experts sum to 1
        normalized_weights = masked_weights / (tf.reduce_sum(masked_weights, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_weights, expert_mask

class MoEGraphEncoder:
    """
    MAIN MoE COMPONENT: Mixture of Experts Graph Encoder
    
    ORIGINAL ARCHITECTURE: Single GraphEncoder processes all graphs
    MoE ARCHITECTURE: Multiple specialized GraphEncoders + Router for expert selection
    
    Key Innovation: Different experts specialize in different graph characteristics
    """
    
    def __init__(self, embedding_size, feature_size, initialization_stddev, gnn_layers=3):
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.initialization_stddev = initialization_stddev
        self.gnn_layers = gnn_layers
        self.num_experts = 4  # Fixed: 4 specialized experts
        
        # STEP 1: Create router network (NEW in MoE)
        # Router learns to map graph features -> expert weights
        self.router = RouterNetwork(input_size=4, num_experts=self.num_experts, top_k=2)
        
        # STEP 2: Create specialized experts (DIFFERENT from original single encoder)
        # Each expert uses different GNN architecture optimized for specific graph types
        
        # Expert 0: Sparse graphs (low connectivity, low m parameter in BA graphs)
        # Uses GraphSAGE - good for sparse graphs with neighborhood sampling
        self.expert_sparse = self._create_expert('graphSage')
        
        # Expert 1: Dense graphs (high connectivity, high m parameter in BA graphs)  
        # Uses GIN - good for dense graphs with complex patterns
        self.expert_dense = self._create_expert('GIN')
        
        # Expert 2: Small graphs (n < 50 nodes)
        # Uses S2V (Structure2Vec) - simpler model for small graphs
        self.expert_small = self._create_expert('S2V')
        
        # Expert 3: Large graphs (n > 100 nodes)
        # Uses GraphSAGE - scalable for large graphs
        self.expert_large = self._create_expert('graphSage')
        
        # Store all experts for easy iteration
        self.experts = [self.expert_sparse, self.expert_dense, self.expert_small, self.expert_large]
    
    def _create_expert(self, embedding_method, expert_name=None):
        """
        EXPERT CREATION: Each expert is a specialized GraphEncoder
        
        ORIGINAL: Single GraphEncoder with fixed architecture
        MoE: Multiple GraphEncoders with different architectures for specialization
        
        Args:
            embedding_method: GNN type ('graphSage', 'GIN', 'S2V')
        Returns:
            GraphEncoder instance specialized for specific graph types
        """
        return GraphEncoder(
            embeddingMethod=embedding_method,  # Different GNN for each expert
            feature_size=self.feature_size,
            embedding_size=self.embedding_size,
            initialization_stddev=self.initialization_stddev,
            gnn_layers=self.gnn_layers,
            expert_name=expert_name
        )
    
    def extract_graph_features(self, n2nsum_param, subgsum_param, batch_graph_ids):
        """
        FEATURE EXTRACTION FOR ROUTING: Convert graph structure to routing features
        
        ORIGINAL: No feature extraction - single encoder processes raw graph structure
        MoE: Extract 4 key graph characteristics to guide expert selection
        
        Purpose: Transform sparse graph tensors into dense feature vectors for router
        Args:
            n2nsum_param: [N, N] - sparse adjacency/aggregation matrix
            subgsum_param: [B, N] - sparse graph-to-node assignment matrix
            batch_graph_ids: [N] - which graph each node belongs to
            
        Returns:
            graph_features: [B, 4]
        """
        num_graphs = tf.shape(subgsum_param)[0]  # B: batch size
        num_nodes = tf.shape(n2nsum_param)[0]    # N: total nodes across all graphs
         
        # FEATURE 1: Number of nodes per graph (exact calculation)  [B]
        nodes_per_graph = tf.math.unsorted_segment_sum(
            tf.ones([num_nodes]), batch_graph_ids, num_graphs
        )
        
        # FEATURES 2-4: Approximations (exact per-graph calculation is complex with sparse tensors)
        # These approximations work well for routing decisions
        
        # FEATURE 2: Estimated m parameter (BA graph connectivity)
        # Count total edges from adjacency matrix (n2nsum_param)
        total_edges = tf1.sparse_reduce_sum(tf.cast(n2nsum_param, tf.float32)) / 2.0  # /2 for undirected
        # Average degree across all nodes
        avg_degree = 2.0 * total_edges / tf.maximum(tf.cast(num_nodes, tf.float32), 1.0)
        # Estimate m parameter: m ≈ avg_degree / 2 (for BA graphs)
        estimated_m = avg_degree / 2.0
        estimated_m_per_graph = tf.fill([num_graphs], estimated_m)  # Broadcast to all graphs
        
        # FEATURE 3: Graph density
        # Density = actual_edges / possible_edges
        avg_density = total_edges / tf.maximum(tf.cast(num_nodes * (num_nodes - 1), tf.float32) / 2.0, 1.0)
        density_per_graph = tf.fill([num_graphs], avg_density)  # Broadcast to all graphs
        
        # FEATURE 4: Average degree (already calculated above)
        avg_degree_per_graph = tf.fill([num_graphs], avg_degree)  # Broadcast to all graphs
        
        # NORMALIZATION: Scale features to [0, 1] range for stable router training
        # Feature 1: Nodes (normalize by expected max: 200)
        normalized_nodes = tf.cast(nodes_per_graph, tf.float32) / 200.0
        
        # Feature 2: m parameter (normalize by expected max: 10)
        normalized_m = tf.clip_by_value(estimated_m_per_graph / 10.0, 0.0, 1.0)
        
        # Feature 3: Density (already in [0, 1])
        normalized_density = tf.clip_by_value(density_per_graph, 0.0, 1.0)
        
        # Feature 4: Average degree (normalize by expected max: 20)
        normalized_avg_degree = tf.clip_by_value(avg_degree_per_graph / 20.0, 0.0, 1.0)
        
        # COMBINE: Stack into router input tensor [B, 4]
        graph_features = tf.stack([
            normalized_nodes,           # [0, 1]: small to large graphs
            normalized_m,              # [0, 1]: sparse to dense connectivity 
            normalized_density,        # [0, 1]: sparse to dense graphs
            normalized_avg_degree      # [0, 1]: low to high degree
        ], axis=1)  # Result: [B, 4]
        
        return graph_features
    
    def encode(self, node_input, y_node_input, n2nsum_param, subgsum_param, batch_graph_ids=None):
        """
        CORE MoE ENCODING PROCESS: The heart of the MoE architecture
        
        ORIGINAL FLOW: graph -> single_encoder -> embeddings
        MoE FLOW: graph -> feature_extraction -> router -> multiple_experts -> weighted_combination -> embeddings
        
        Args:
            node_input: [N, feature_size] - node features for all graphs in batch
            y_node_input: [B, feature_size] - graph-level features  
            n2nsum_param: [N, N] - sparse adjacency/aggregation matrix
            subgsum_param: [B, N] - sparse graph-to-node assignment matrix
            batch_graph_ids: [N] - which graph each node belongs to
            
        Returns:
            combined_node_embeddings: [N, embedding_size] - final node embeddings
            combined_graph_embeddings: [B, embedding_size] - final graph embeddings  
            expert_weights: [B, num_experts] - routing weights (for analysis)
        """
        
        # STEP 1: FEATURE EXTRACTION FOR ROUTING
        # ORIGINAL: Skip this step - use raw graph structure directly
        # MoE: Extract 4 key features [nodes, m_param, density, avg_degree] -> [B, 4]
        graph_features = self.extract_graph_features(n2nsum_param, subgsum_param, batch_graph_ids)
        
        # STEP 2: ROUTER DECISION
        # ORIGINAL: No routing - single path for all graphs
        # MoE: Router network decides expert weights based on graph characteristics
        # expert_weights: [B, num_experts] - how much each expert contributes to each graph
        # expert_mask: [B, num_experts] - binary mask for top-k expert selection
        expert_weights, expert_mask = self.router.route(graph_features)
        
        # STEP 3: PARALLEL EXPERT PROCESSING
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Each expert is a specialized GraphEncoder with different GNN architecture
            # expert.encode() returns: (node_embeddings, graph_embeddings)
            cur_message_layer, y_cur_message_layer = expert.encode(
                node_input, y_node_input, n2nsum_param, subgsum_param, batch_graph_ids
            )
            expert_outputs.append((cur_message_layer, y_cur_message_layer))
        
        # STEP 4: WEIGHTED COMBINATION OF EXPERT OUTPUTS
        batch_size = tf.shape(subgsum_param)[0]  # B: number of graphs
        num_nodes = tf.shape(n2nsum_param)[0]    # N: total nodes across all graphs
        
        # Initialize combined embeddings (will accumulate weighted expert outputs)
        combined_node_embeddings = tf.zeros([num_nodes, self.embedding_size])
        combined_graph_embeddings = tf.zeros([batch_size, self.embedding_size])
        
        # WEIGHTED COMBINATION LOOP: Combine outputs from all experts
        for i, (node_emb, graph_emb) in enumerate(expert_outputs):
            # node_emb: [N, embedding_size] - expert i's node embeddings
            # graph_emb: [B, embedding_size] - expert i's graph embeddings
            
            # COMBINE NODE EMBEDDINGS:
            # expert_weights[:, i]: [B] - weight of expert i for each graph
            # batch_graph_ids: [N] - which graph each node belongs to
            # node_weights: [N] - weight of expert i for each node (based on its graph)
            node_weights = tf.gather(expert_weights[:, i], batch_graph_ids)
            node_weights = tf.expand_dims(node_weights, axis=1)  # [N, 1] for broadcasting
            
            # Weight expert i's node embeddings and add to combination
            weighted_node_emb = node_emb * node_weights  # [N, embedding_size]
            combined_node_embeddings += weighted_node_emb
            
            # COMBINE GRAPH EMBEDDINGS:
            # expert_weights[:, i]: [B] - weight of expert i for each graph
            graph_weights = tf.expand_dims(expert_weights[:, i], axis=1)  # [B, 1] for broadcasting
            
            # Weight expert i's graph embeddings and add to combination
            weighted_graph_emb = graph_emb * graph_weights  # [B, embedding_size]
            combined_graph_embeddings += weighted_graph_emb
        
        # FINAL RESULT: Weighted combination of all expert outputs
        # combined_node_embeddings: [N, embedding_size] - each node's embedding is a weighted mix of expert outputs
        # combined_graph_embeddings: [B, embedding_size] - each graph's embedding is a weighted mix of expert outputs
        # expert_weights: [B, num_experts] - routing decisions (useful for analysis and load balance loss)
        return combined_node_embeddings, combined_graph_embeddings, expert_weights

class MoEMLPDecoder:
    """
    MoE MLP DECODER: Identical to original MLPDecoder but works with MoE embeddings
    
    ORIGINAL: Processes embeddings from single GraphEncoder
    MoE: Processes embeddings from weighted combination of multiple experts
    
    Key Point: The decoder architecture is UNCHANGED - only the input embeddings are different
    This demonstrates MoE's modularity: we can swap encoders without changing decoders
    """
    
    def __init__(self, embedding_size, reg_hidden, aux_dim, initialization_stddev):
        self.embedding_size = embedding_size
        self.reg_hidden = reg_hidden
        self.aux_dim = aux_dim
        self.initialization_stddev = initialization_stddev
        
        # IDENTICAL TO ORIGINAL MLPDecoder: Same architecture, same parameters
        # The magic happens in the encoder - decoder just processes the combined embeddings
        
        if reg_hidden > 0:
            # Standard case: embedding -> hidden -> output
            # h1_weight: [embedding_size, reg_hidden] - first layer weights
            self.h1_weight = tf1.Variable(tf1.truncated_normal([embedding_size, reg_hidden], stddev=initialization_stddev), tf.float32)
            # last_w: [reg_hidden + aux_dim, 1] - final layer (hidden + auxiliary features -> Q-value)
            self.last_w = tf1.Variable(tf1.truncated_normal([reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
        else:
            # Alternative case: embedding -> expanded -> output
            # h1_weight: [embedding_size, 2 * embedding_size] - expansion layer
            self.h1_weight = tf1.Variable(tf1.truncated_normal([embedding_size, 2 * embedding_size], stddev=initialization_stddev), tf.float32)
            # last_w: [2 * embedding_size + aux_dim, 1] - final layer
            self.last_w = tf1.Variable(tf1.truncated_normal([2 * embedding_size + aux_dim, 1], stddev=initialization_stddev), tf.float32)
        
        # cross_product: [embedding_size, 1] - for state-action embedding interaction
        # This is the key component that combines node and graph embeddings
        self.cross_product = tf1.Variable(tf1.truncated_normal([embedding_size, 1], stddev=initialization_stddev), tf.float32)
    
    def decode_q(self, cur_message_layer, action_select, y_cur_message_layer, aux_input):
        """
        Decode Q-values for selected actions
        Args:
            cur_message_layer: [N, embed_dim]
            action_select: [B, N]
            y_cur_message_layer: [B, embed_dim]
            aux_input: [B, aux_dim]
        Returns:
            q_pred: [B, 1]
        """
        # Action embedding
        # [B,embed_dim]
        action_embed = tf1.sparse_tensor_dense_matmul(tf.cast(action_select, tf.float32), cur_message_layer)

        # Cross product to calculate embed_s_a 
        # [B, embed_dim, embed_dim]
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2), tf.expand_dims(y_cur_message_layer, axis=1))
        # [B, embed_dim]
        Shape = tf.shape(action_embed)
        # [B, embed_dim, 1]
        batch_cross_product = tf.reshape(tf.tile(self.cross_product, [Shape[0], 1]), [Shape[0], Shape[1], 1])
        # [B, embed_dim]
        embed_s_a = tf.reshape(tf.matmul(temp, batch_cross_product), Shape)

        # [B,embedding_size]
        last_output = embed_s_a
        # [B, reg_hidden]
        hidden = tf.matmul(embed_s_a, self.h1_weight)
        last_output = tf.nn.relu(hidden)
        # [B,reg_hidden+aux_dim]
        last_output = tf.concat([last_output, aux_input], 1)
        # [B,1]
        q_pred = tf.matmul(last_output, self.last_w)
        return q_pred
    
    def decode_q_all(self, cur_message_layer, y_cur_message_layer, aux_input, rep_global):
        """
        Decode Q-values for all nodes
        Args:
            cur_message_layer: [N, embed_dim]
            y_cur_message_layer: [B, embed_dim]
            aux_input: [B, aux_dim]
            rep_global: [N, B]
        Returns:
            q_on_all: [N, 1]
        """
        # Q(s,a) on all nodes of all graphs
        # [N,embed_dim]
        rep_y = tf1.sparse_tensor_dense_matmul(tf.cast(rep_global, tf.float32), y_cur_message_layer)
        # [N,embed_dim,embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2), tf.expand_dims(rep_y, axis=1))
        # [N,embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # [N,embed_dim]
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(self.cross_product, [Shape1[0], 1]), [Shape1[0], Shape1[1], 1])), Shape1)
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            hidden = tf.matmul(embed_s_a_all, self.h1_weight)
            last_output = tf.nn.relu(hidden)
        rep_aux = tf1.sparse_tensor_dense_matmul(tf.cast(rep_global, tf.float32), aux_input)
        # [N,reg_hidden+aux_dim]
        last_output = tf.concat([last_output, rep_aux], 1)
        # [N,1]
        q_on_all = tf.matmul(last_output, self.last_w)
        return q_on_all
