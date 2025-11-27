
import numpy as np
import tensorflow as tf
from inits import glorot
from inits import normalize_features


conv1d = tf.layers.conv1d

def attn_head(features, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0,
              return_coef=True):
    
    with tf.name_scope('my_attn'):
        
        seq_fts = seq
        
        hidden_size=seq.shape[2].value
        attention_size=8
        feature_size=features.shape[2].value
        
        w1_attention=tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        w2_attention=tf.Variable(tf.random_normal([feature_size, attention_size], stddev=0.1))
        b_attention = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v_attention=tf.Variable(tf.random_normal([attention_size,1], stddev=0.1))
        
        val_mid=tf.tanh(tf.add(tf.matmul(seq_fts[0],w1_attention),tf.matmul(features[0],w2_attention)))
        val_mid=tf.add(val_mid,b_attention)
        logits=tf.matmul(val_mid,v_attention)
        logits=logits+tf.transpose(logits)
        logits=logits[tf.newaxis]
        
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  

        vals = tf.matmul(coefs, seq_fts)       
        ret = tf.contrib.layers.bias_add(vals)
        
        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)  # activation

def gcn_layer(embed, inter_mat):
    
    node_size=embed.shape[1].value
    embed_size=embed.shape[2].value
    latent_factor_size = 64
    #with tf.compat.v1.variable_scope("enco_second"):
    with tf.variable_scope("enco_second"):
        weight4 = glorot([embed_size,latent_factor_size])
        weight5 = glorot([node_size,latent_factor_size])
    
    con = tf.matmul(inter_mat[0], embed[0])
    hidden = tf.add(tf.matmul(con,weight4),weight5)
    hidden = hidden[tf.newaxis]
    
    return tf.nn.relu(hidden)
    
def SimpleAttLayer(embed_list, features, inputs, attention_size, time_major=False, return_alphas=False, fea_size_v=8):   
    
    feature_size = features.shape[2].value  
    hidden_size = embed_list[0].shape[2].value   
    

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))   
    v_omega = tf.Variable(tf.random_normal([feature_size, attention_size], stddev=0.1)) #1437
    u_omega = tf.Variable(tf.random_normal([attention_size,1], stddev=0.1))  
    
    
    F1 = np.loadtxt("../data/drug_features.txt") #1373x1373
    F2 = np.loadtxt("../data/microbe_features.txt")
    
    feat = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((F2,np.zeros(shape=(F2.shape[0],F1.shape[0]),dtype=int)))))
    feat = normalize_features(feat) 
    feat = tf.constant(feat,tf.float32)
   
    
    alphas_temp = []
    for adj in embed_list:
        val=tf.tanh(tf.add(tf.matmul(adj[0],w_omega),tf.matmul(feat,v_omega)))
        val_final = tf.matmul(val,u_omega)
        alphas_temp.append(val_final)
    alphas = tf.concat(alphas_temp,axis=-1)
    alphas = tf.nn.softmax(alphas)   
    
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)   
     
    if not return_alphas:
        return output
    else:
        return output, alphas
    
def gcn(seq, weight_val1, weight_val2, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0,
              return_coef=True):
   
    weight1=weight_val1
    weight2=weight_val2
    
    with tf.name_scope('my_attn'):
        
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  
        
        
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])   
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)                                                                      

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
        
        hidden=tf.matmul(coefs[0],seq_fts[0])
        hidden=tf.add(tf.matmul(hidden,weight1),weight2)
        hidden=hidden[np.newaxis]

        if return_coef:
            return activation(hidden), coefs
        else:
            return activation(hidden)  # activation    



        
        