import numpy as np
import tensorflow as tf

import layers
from base_gattn import BaseGAttN
from inits import glorot


class HeteGAT(BaseGAttN):
    def encoder(self,inputs_list, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, inter_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=512):
        coefs=tf.constant([1,2])
        att_val=tf.constant([1,2])
        embed_list_temp=[]
        embed_list = []
        for inputs, bias_mat, inter_mat in zip(inputs_list, bias_mat_list, inter_mat_list):
            attns = []
            for _ in range(n_heads[0]):

                inputs = layers.gcn_layer(inputs, inter_mat)  #GCN

                attns_temp, coefs = layers.attn_head(inputs_list[0], inputs, bias_mat=bias_mat,   #GAT
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop)
                attns.append(attns_temp)
            h_1 = tf.concat(attns, axis=-1)
            embed_list_temp.append(h_1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(embed_list_temp, inputs_list[0], multi_embed, mp_att_size,
                                                     return_alphas=True)

        return final_embed, att_val, embed_list_temp

    def decoder(self,embed):
        embed_size = embed.shape[1].value
        #with tf.compat.v1.variable_scope("deco"):
        with tf.variable_scope("deco",reuse=tf.AUTO_REUSE):
             weight3 = glorot([embed_size,embed_size])
        U = embed[0:1373,:]
        V = embed[1373:,:]

        U = self.apply_neural_completion(U)
        V = self.apply_neural_completion(V)

        logits = tf.matmul(U, tf.transpose(V))
        #logits = tf.matmul(tf.matmul(U,weight3),tf.transpose(V))
        logits = tf.reshape(logits,[-1,1])

        return tf.nn.relu(logits)

    def apply_neural_completion(self, input_tensor):
        with tf.variable_scope("neural_completion", reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.dense(input_tensor, units=256, activation=tf.nn.relu, name='layer_1')
            layer_2 = tf.layers.dense(layer_1, units=128, activation=tf.nn.relu, name='layer_2')
            layer_3 = tf.layers.dense(layer_2, units=64, activation=tf.nn.relu, name='layer_3')
        return layer_3
