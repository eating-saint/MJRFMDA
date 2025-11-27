

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf           
            
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr):
    
    labels = np.loadtxt("../data/adj.txt")
    
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(1373,173)).toarray()
    logits_test = logits_test.reshape([-1,1])  
    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(1373,173)).toarray()
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])
            
    F1 = np.loadtxt("../data/drug_features.txt")
    F2 = np.loadtxt("../data/microbe_features.txt")
    

    features = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[0]),dtype=int), F2))))
    features = normalize_features(features)
    
    interaction=[]
    rownetworks=[]
    P1 = sio.loadmat('../data/net1.mat')
    P2 = sio.loadmat('../data/net2.mat')
    P3 = sio.loadmat('../data/net3.mat')
    
    #P1 = sio.loadmat('D:/anaconda3/work/DGATMDA/net123.mat')
    #P1_v = P1['net123']
    
    P1_v = P1['interaction']
    P2_v = P2['net2']
    P3_v = P3['net3_sub']
    
    P_1 = np.vstack((np.hstack((np.zeros(shape=(1373,1373),dtype=int),P1_v)),np.hstack((P1_v.transpose(),np.zeros(shape=(173,173),dtype=int)))))
    P_2 = np.vstack((np.hstack((np.zeros(shape=(1373,1373),dtype=int),P2_v)),np.hstack((P2_v.transpose(),np.zeros(shape=(173,173),dtype=int)))))
    P_3 = np.vstack((np.hstack((np.zeros(shape=(1373,1373),dtype=int),P3_v)),np.hstack((P3_v.transpose(),np.zeros(shape=(173,173),dtype=int)))))
    
    interaction1 = preprocess_adj(P_1)
    interaction2 = preprocess_adj(P_2)
    interaction3 = preprocess_adj(P_3)
   
    interaction.append(interaction1)
    interaction.append(interaction2)
    interaction.append(interaction3)
   
    P_1 = P_1+np.eye(1546)
    P_2 = P_2+np.eye(1546)
    P_3 = P_3+np.eye(1546)
    
    rownetworks.append(P_1)
    rownetworks.append(P_2)
    rownetworks.append(P_3)
    
    truefeatures_list = [features, features, features]
    
    return interaction, rownetworks, truefeatures_list, logits_train, logits_test, train_mask, test_mask, labels

def generate_mask(labels,N):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((1*N,2)) 
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            label_neg[num,0]=a
            label_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])  
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()  
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    #initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

