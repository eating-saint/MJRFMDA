import numpy as np
from collections import defaultdict
import random


def cross_validation(intMat, seeds, cv=0, num=5):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size) 
        step = index.size/num   
        for i in range(num):
            if i < num-1:
                ii = index[int(i*step):int((i+1)*step)]
            else:
                ii = index[int(i*step):]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)  
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)   
            x, y = test_data[:, 0], test_data[:, 1]  
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)     
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))   
    return cv_data
                
def generate_negative_sample(adj,interaction,N):
    num = 0
    mask = np.zeros(interaction.shape)
    test_neg=np.zeros((1*N,2))
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if interaction[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg                

def generate_train_negtive_samples(W, intMat):
    
    [x_all, y_all] = np.where(W==1)
    input_data = W*intMat
    [x_pos, y_pos] = np.where(input_data==1)
    [x_neg, y_neg] = np.where((W==1) & (input_data==0))
    
    
    num_pos = x_pos.size  #1236
    reorder = np.arange(len(x_neg))
    np.random.shuffle(reorder)
    x_neg_mask = x_neg[reorder[0:num_pos]]
    y_neg_mask = y_neg[reorder[0:num_pos]]
    
    mask_neg = np.zeros(W.shape)
    mask_neg [x_neg_mask, y_neg_mask] = 1  #
    
    label_neg = intMat[x_neg_mask,y_neg_mask]
    
    mask_neg = np.reshape(mask_neg, [-1,1])
    label_neg = np.reshape(label_neg, [-1,1])
    
    
    return  mask_neg, label_neg