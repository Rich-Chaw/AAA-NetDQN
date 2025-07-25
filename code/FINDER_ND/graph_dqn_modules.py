import tensorflow as tf
import tensorflow.compat.v1 as tf1

class GraphEncoder:
    def __init__(self, embedding_size, initialization_stddev, max_bp_iter, embeddingMethod, aggregatorID):
        self.embedding_size = embedding_size
        self.initialization_stddev = initialization_stddev
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aggregatorID = aggregatorID
        # Weights
        self.w_n2l = tf.Variable(tf1.truncated_normal([2, embedding_size], stddev=initialization_stddev), tf.float32)
        self.p_node_conv = tf.Variable(tf1.truncated_normal([embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 1:
            self.p_node_conv2 = tf.Variable(tf1.truncated_normal([embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
            self.p_node_conv3 = tf.Variable(tf1.truncated_normal([2*embedding_size, embedding_size], stddev=initialization_stddev), tf.float32)
        else:
            self.p_node_conv2 = None
            self.p_node_conv3 = None

    def encode(self, n2nsum_param, subgsum_param):
        # Placeholders are passed in
        nodes_size = tf.shape(n2nsum_param)[0]
        y_nodes_size = tf.shape(subgsum_param)[0]
        node_input = tf.cast(tf.ones((nodes_size,2)),tf.float32)
        y_node_input = tf.cast(tf.ones((y_nodes_size,2)),tf.float32)
        input_message = tf.matmul(node_input, self.w_n2l)
        y_input_message = tf.matmul(y_node_input, self.w_n2l)
        cur_message_layer = tf.nn.l2_normalize(tf.nn.relu(input_message), axis=1)
        y_cur_message_layer = tf.nn.l2_normalize(tf.nn.relu(y_input_message), axis=1)
        lv = 0
        while lv < self.max_bp_iter:
            lv += 1
            n2npool = tf1.sparse_tensor_dense_matmul(tf.cast(n2nsum_param,tf.float32), cur_message_layer)
            node_linear = tf.matmul(n2npool, self.p_node_conv)
            y_n2npool = tf1.sparse_tensor_dense_matmul(tf.cast(subgsum_param,tf.float32), cur_message_layer)
            y_node_linear = tf.matmul(y_n2npool, self.p_node_conv)
            if self.embeddingMethod == 0:
                merged_linear = tf.add(node_linear, input_message)
                cur_message_layer = tf.nn.relu(merged_linear)
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), self.p_node_conv2)
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, self.p_node_conv3))
                
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), self.p_node_conv2)
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, self.p_node_conv3))

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
        last_output = tf.concat([last_output, rep_aux], 1)
        # [N,1]
        q_on_all = tf.matmul(last_output, self.last_w)
        return q_on_all