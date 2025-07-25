import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

import numpy as np
import networkx as nx
import random

class GraphEncoder:
    def __init__(self, embeddingMethod, feature_size, embedding_size, initialization_stddev, gnn_layers=3):
        self.embeddingMethod = embeddingMethod
        self.embedding_size = embedding_size
        self.initialization_stddev = initialization_stddev
        self.gnn_layers = gnn_layers
        self.feature_size = feature_size

        # Weights
        # GIN, originally in GIN
        if embeddingMethod == 'GIN':
            # Create MLP weights for each GIN layer
            self.gin_mlps = []
            for i in range(gnn_layers):
                w1 = tf.Variable(tf1.truncated_normal([embedding_size if i > 0 else feature_size, embedding_size], stddev=initialization_stddev), tf.float32)
                b1 = tf.Variable(tf.zeros([embedding_size]), tf.float32)
                w2 = tf.Variable(tf1.truncated_normal([embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
                b2 = tf.Variable(tf.zeros([embedding_size]), tf.float32)
                self.gin_mlps.append((w1, b1, w2, b2))
            self.epsilons = [tf.Variable(0.0, dtype=tf.float32) for _ in range(gnn_layers)]
        
        # graphSage or S2V ,originally in FINDER
        self.w_n2l = tf.Variable(tf1.truncated_normal([feature_size, embedding_size], stddev=initialization_stddev), tf.float32)
        # W_1 
        self.p_node_conv = tf.Variable(tf1.truncated_normal([embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 'graphSage':
            # W_2
            self.p_node_conv2 = tf.Variable(tf1.truncated_normal([embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
            # W_3
            self.p_node_conv3 = tf.Variable(tf1.truncated_normal([2*embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
        elif embeddingMethod == 'S2V':
            self.p_node_conv2 = None
            self.p_node_conv3 = None

    def encode(self, node_input, y_node_input, n2nsum_param, subgsum_param, batch_graph_ids=None):
        '''
            node_input: [N,feature_size]
            y_node_input: [B,feature_size]
            n2nsum_param: [N,N]
            subgsum_param: [B,N]
            Note in GraphDQN: 
            B = num_graphs (in a batch)
            N = B * num_nodes_per_graph, however: graphs have variable numbers of nodes
        '''
        num_graphs = tf.shape(subgsum_param)[0]       # num graphs in a batch
        num_nodes = tf.shape(n2nsum_param)[0]         # total batch num of a graph
        num_nodes_per_graph = num_nodes // num_graphs  # num nodes per graph

        # GIN, originally in GIN
        if self.embeddingMethod == 'GIN':
            # [N, feature_size]
            h = node_input  
            for i in range(self.gnn_layers):
                # Aggregate neighbors
                agg = tf.sparse.sparse_dense_matmul(n2nsum_param, h)
                # Add self-loop with learnable epsilon
                h_self = (1.0 + self.epsilons[i]) * h
                h_agg = agg + h_self
                # MLP: two-layer with ReLU
                w1, b1, w2, b2 = self.gin_mlps[i]
                h_agg = tf.nn.relu(tf.matmul(h_agg, w1) + b1)
                h = tf.matmul(h_agg, w2) + b2
                h = tf.nn.relu(h)
            cur_message_layer = h  # [N, embedding_size]
            # Use TensorFlow ops for batch_graph_ids
            y_cur_message_layer = tf.math.unsorted_segment_sum(cur_message_layer, batch_graph_ids, num_graphs)

        # graphSage or S2V ,originally in FINDER
        if self.embeddingMethod == 'graphSage' or self.embeddingMethod == 'S2V':
            # [N,embedding_dim]
            input_message = tf.matmul(node_input, self.w_n2l)
            # [B,embedding_dim]
            y_input_message = tf.matmul(y_node_input, self.w_n2l)
            # [N,embedding_dim] node_embedding
            cur_message_layer = tf.nn.l2_normalize(tf.nn.relu(input_message), axis=1)
            # [B,embedding_dim] graph_embedding = virtual_node_embedding virtual_node_embedding = sum(node_embedding)
            # i.e.  assume a virtual node connect to all nodes in the graph
            y_cur_message_layer = tf.nn.l2_normalize(tf.nn.relu(y_input_message), axis=1)
            
            for i in range(self.gnn_layers):
                n2npool = tf1.sparse_tensor_dense_matmul(tf.cast(n2nsum_param,tf.float32), cur_message_layer)
                node_linear = tf.matmul(n2npool, self.p_node_conv)
                y_n2npool = tf1.sparse_tensor_dense_matmul(tf.cast(subgsum_param,tf.float32), cur_message_layer)
                y_node_linear = tf.matmul(y_n2npool, self.p_node_conv)
                if self.embeddingMethod == 'graphSage':
                    cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), self.p_node_conv2)
                    merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                    cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, self.p_node_conv3))
                    
                    y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), self.p_node_conv2)
                    y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                    y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, self.p_node_conv3))

                elif self.embeddingMethod == 'S2V':
                    merged_linear = tf.add(node_linear, input_message)
                    cur_message_layer = tf.nn.relu(merged_linear)
                    y_merged_linear = tf.add(y_node_linear, y_input_message)
                    y_cur_message_layer = tf.nn.relu(y_merged_linear)

                cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
                y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        return cur_message_layer, y_cur_message_layer

class MLPDecoder:
    def __init__(self, embedding_size, reg_hidden, aux_dim, initialization_stddev):
        self.embedding_size = embedding_size
        self.reg_hidden = reg_hidden
        self.aux_dim = aux_dim
        self.initialization_stddev = initialization_stddev
        if reg_hidden > 0:
            self.h1_weight = tf.Variable(tf1.truncated_normal([embedding_size, reg_hidden], stddev=initialization_stddev), tf.float32)
            self.last_w = tf.Variable(tf1.truncated_normal([reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
        else:
            self.h1_weight = tf.Variable(tf1.truncated_normal([embedding_size, 2 * embedding_size], stddev=initialization_stddev), tf.float32)
            self.last_w = tf.Variable(tf1.truncated_normal([2 * embedding_size + aux_dim, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = tf.Variable(tf1.truncated_normal([embedding_size, 1], stddev=initialization_stddev), tf.float32)

    def decode_q(self, cur_message_layer,action_select, y_cur_message_layer, aux_input):
        '''
        cur_message_layer: [N, embed_dim]
        action_select: [B, N]
        y_cur_message_layer: [B, embed_dim]
        aux_input: [B, aux_dim]

        return: [B, 1]
        '''
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
        '''
        cur_message_layer: [N, embed_dim]
        y_cur_message_layer: [B, embed_dim]
        aux_input: [B, aux_dim]
        rep_global: [N, B]

        return: [N, 1]
        '''
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




class MockPrepareBatchGraph:
    def __init__(self, aggregatorID):
        self.aggregatorID = aggregatorID
        # These will be set in SetupTrain
        self.act_select = None
        self.rep_global = None
        self.n2nsum_param = None
        self.laplacian_param = None
        self.subgsum_param = None
        self.aux_feat = None
        self.batch_graph_ids = None

    def SetupTrain(self, idxes, g_list, covered, actions):
        # Assume all graphs have the same number of nodes for simplicity
        num_graphs = len(idxes)
        num_nodes = g_list[0].num_nodes
        total_nodes = num_graphs * num_nodes

        # 1. n2nsum_param: block-diagonal adjacency/aggregation matrix
        n2nsum_indices = []
        n2nsum_values = []
        for g_idx, g in enumerate(g_list):
            offset = g_idx * num_nodes
            adj = np.zeros((num_nodes, num_nodes))
            for u, v in g.edge_list:
                adj[u, v] = 1
                adj[v, u] = 1  # undirected
            for i in range(num_nodes):
                neighbors = np.where(adj[i] != 0)[0]
                for j in neighbors:
                    if self.aggregatorID == 0:  # sum
                        val = 1.0
                    elif self.aggregatorID == 1:  # mean
                        val = 1.0 / len(neighbors) if len(neighbors) > 0 else 0.0
                    elif self.aggregatorID == 2:  # GCN
                        deg_i = len(neighbors)
                        deg_j = len(np.where(adj[j] != 0)[0])
                        val = 1.0 / (np.sqrt((deg_i + 1) * (deg_j + 1)))
                    else:
                        raise ValueError("Unknown aggregatorID")
                    n2nsum_indices.append([offset + i, offset + j])
                    n2nsum_values.append(val)
        self.n2nsum_param = tf1.SparseTensorValue(
            indices=np.array(n2nsum_indices, dtype=np.int64),
            values=np.array(n2nsum_values, dtype=np.float32),
            dense_shape=(total_nodes, total_nodes)
        )

        # 2. subgsum_param: assigns nodes to graphs (one-hot per graph, block-diagonal)
        subgsum_indices = []
        subgsum_values = []
        for g_idx in range(num_graphs):
            for n in range(num_nodes):
                subgsum_indices.append([g_idx, g_idx * num_nodes + n])
                subgsum_values.append(1.0)
        self.subgsum_param = tf1.SparseTensorValue(
            indices=np.array(subgsum_indices, dtype=np.int64),
            values=np.array(subgsum_values, dtype=np.float32),
            dense_shape=(num_graphs, total_nodes)
        )

        # 3. act_select: one-hot for the action taken in each graph
        act_select_indices = []
        act_select_values = []
        for g_idx, a in enumerate(actions):
            act_select_indices.append([g_idx, g_idx * num_nodes + a])
            act_select_values.append(1.0)
        self.act_select = tf1.SparseTensorValue(
            indices=np.array(act_select_indices, dtype=np.int64),
            values=np.array(act_select_values, dtype=np.float32),
            dense_shape=(num_graphs, total_nodes)
        )

        # 4. rep_global: assigns each node to its graph (one-hot per node, per graph)
        rep_global_indices = []
        rep_global_values = []
        for g_idx in range(num_graphs):
            for n in range(num_nodes):
                rep_global_indices.append([g_idx * num_nodes + n, g_idx])
                rep_global_values.append(1.0)
        self.rep_global = tf1.SparseTensorValue(
            indices=np.array(rep_global_indices, dtype=np.int64),
            values=np.array(rep_global_values, dtype=np.float32),
            dense_shape=(total_nodes, num_graphs)
        )

        # 5. aux_feat: simple features (e.g., fraction of covered nodes)
        aux_feat = []
        for g_idx, cov in enumerate(covered):
            frac_covered = len(cov) / num_nodes
            aux_feat.append([frac_covered, 0, 0, 0])  # pad to 4 dims
        self.aux_feat = np.array(aux_feat, dtype=np.float32)

        # Create batch_graph_ids for GIN encoder
        batch_graph_ids = []
        for g_idx, g in enumerate(g_list):
            batch_graph_ids.extend([g_idx] * g.num_nodes)
        self.batch_graph_ids = np.array(batch_graph_ids, dtype=np.int32)
        
        # Print for explanation
        print("n2nsum_param (indices):", self.n2nsum_param.indices)
        print("n2nsum_param (values):", self.n2nsum_param.values)
        print("subgsum_param (indices):", self.subgsum_param.indices)
        print("act_select (indices):", self.act_select.indices)
        print("rep_global (indices):", self.rep_global.indices)
        print("aux_feat:", self.aux_feat)
        print("batch_graph_ids:", self.batch_graph_ids)

        


# --- Add main function for demonstration ---
def main():
    # 1. Generate 2 Barabasi-Albert graphs with 5 nodes each
    num_graphs = 2
    num_nodes = 5 # num_nodes per graph
    m = 2
    graphs = [nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed) for seed in range(num_graphs)]

    class MockGraph:
        def __init__(self, nxg):
            self.num_nodes = nxg.number_of_nodes()
            self.num_edges = nxg.number_of_edges()
            self.edge_list = list(nxg.edges())
            self.adj_list = [list(nxg.neighbors(i)) for i in range(self.num_nodes)]
    g_list = [MockGraph(g) for g in graphs]

    # Example covered and actions
    covered = [[0, 2], [1, 3, 4]]
    actions = [3, 0]
    idxes = np.arange(num_graphs)

    # Prepare batch
    # aggregatorID 0: sum, 1: mean, 2: GCN
    prepareBatchGraph = MockPrepareBatchGraph(aggregatorID=1)  
    prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
    n2nsum_param = prepareBatchGraph.n2nsum_param
    subgsum_param = prepareBatchGraph.subgsum_param    
    action_select = prepareBatchGraph.act_select
    rep_global = prepareBatchGraph.rep_global
    aux_input= prepareBatchGraph.aux_feat
    batch_graph_ids = prepareBatchGraph.batch_graph_ids

    # initialize node feature
    # N: number of nodes (of all graphs in a batch)
    nodes_size = num_graphs * num_nodes
    # B: batch_size (number of graphs in a batch)
    y_nodes_size = num_graphs

    feature_size = 2 # 2 features for node and graph

    # X: [N,feature_size] node feature, initialized with 1
    node_input = tf.cast(tf.ones((nodes_size,feature_size)),tf.float32)
    # Y: [B,feature_size] graph feature, initialized with 1
    y_node_input = tf.cast(tf.ones((y_nodes_size,feature_size)),tf.float32)

    # 4. Initialize encoder/decoder and run a forward pass
    embedding_size = 64
    initialization_stddev = 0.01
    aux_dim = 4
    reg_hidden = 32
    encoder = GraphEncoder(
        embeddingMethod='GIN',
        feature_size=feature_size,
        embedding_size=embedding_size,
        initialization_stddev=initialization_stddev,
        gnn_layers=3
    )
    decoder = MLPDecoder(
        embedding_size=embedding_size,
        reg_hidden=reg_hidden,
        aux_dim=aux_dim,
        initialization_stddev=initialization_stddev
    )



    # Define placeholders for all sparse tensors at the top of your `main()` function:
    n2nsum_param_ph = tf1.sparse_placeholder(tf.float32, name="n2nsum_param")
    subgsum_param_ph = tf1.sparse_placeholder(tf.float32, name="subgsum_param")
    action_select_ph = tf1.sparse_placeholder(tf.float32, name="action_select")
    rep_global_ph = tf1.sparse_placeholder(tf.float32, name="rep_global")
    aux_input_ph = tf1.placeholder(tf.float32, shape=(None, aux_dim), name="aux_input")
    batch_graph_ids_ph = tf1.placeholder(tf.int32, shape=[None], name='batch_graph_ids')

    # Build computation graph
    cur_message_layer, y_cur_message_layer = encoder.encode(node_input,y_node_input, n2nsum_param_ph, subgsum_param_ph, batch_graph_ids_ph)
    q_pred = decoder.decode_q(cur_message_layer, action_select_ph, y_cur_message_layer, aux_input_ph)
    q_on_all = decoder.decode_q_all(cur_message_layer, y_cur_message_layer, aux_input_ph, rep_global_ph)

    # 5. Minimal training step
    target = np.ones((num_graphs, 1), dtype=np.float32)
    loss = tf.reduce_mean(tf.square(q_pred - target))
    train_step = tf1.train.AdamOptimizer(0.001).minimize(loss)

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        feed_dict = {
            n2nsum_param_ph: n2nsum_param,
            subgsum_param_ph: subgsum_param,
            action_select_ph: action_select,
            rep_global_ph: rep_global,
            aux_input_ph: aux_input,
            batch_graph_ids_ph: batch_graph_ids,
        }
        # In real code, feed_dict would map placeholders to the above mock sparse tensors
        # Here, just run one step for demonstration
        l, _ = sess.run([loss, train_step], feed_dict=feed_dict)
        print(f"Dummy training step completed. Loss: {l}")

if __name__ == "__main__":
    main()

