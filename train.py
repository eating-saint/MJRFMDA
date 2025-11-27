import time
import numpy as np
import tensorflow as tf

from gat import HeteGAT
import process
from inits import load_data
from inits import generate_mask
from inits import test_negative_sample
from metrics import masked_accuracy
from metrics import custom_loss
from metrics import ROC
from metrics import calculate_accuracy

def train(train_arr, test_arr, mask_neg, label_neg):
    
  # training params
  batch_size = 1
  nb_epochs = 200
  lr = 0.005  
  l2_coef = 5e-4
  weight_decay=5e-4
  hid_units = [8]
  n_heads = [1, 1]  
  residual = False
  nonlinearity = tf.nn.elu
  model = HeteGAT()
  alpha = 0.2

  #print('Dataset: ' + dataset)
  print('----- Opt. hyperparams -----')
  print('lr: ' + str(lr))
  print('l2_coef: ' + str(l2_coef))
  print('----- Archi. hyperparams -----')
  print('nb. layers: ' + str(len(hid_units)))
  print('nb. units per layer: ' + str(hid_units))
  print('nb. attention heads: ' + str(n_heads))
  print('residual: ' + str(residual))
  print('nonlinearity: ' + str(nonlinearity))

  interaction_list, adj_list, fea_list, y_train, y_test, train_mask, test_mask, labels = load_data(train_arr, test_arr)  

  print(fea_list[0].shape)

  nb_nodes = fea_list[0].shape[0]
  ft_size = fea_list[0].shape[1]

  fea_list = [fea[np.newaxis] for fea in fea_list]  
  adj_list = [adj[np.newaxis] for adj in adj_list]
  interaction_list = [inter[np.newaxis] for inter in interaction_list]

  biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]  

  print('build graph...')
  with tf.Graph().as_default():    
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        inter_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='inter_in_{}'.format(i))
                        for i in range(len(interaction_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(237529, batch_size), 
                                name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(237529,batch_size),
                                name='msk_in')
        neg_msk = tf.placeholder(dtype=tf.int32, shape=(237529,batch_size),
                                name='neg_msk')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    final_embedding, att_val, embed_list = model.encoder(ftr_in_list, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       inter_mat_list=inter_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)
    logits = model.decoder(final_embedding)
    
    # cal masked_loss
    loss = masked_accuracy(logits, lbl_in, msk_in, neg_msk, alpha)

    para_neural_completion = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="neural_completion")
    l2_loss_neural_completion = tf.add_n([tf.nn.l2_loss(var) for var in para_neural_completion])
    loss += weight_decay * l2_loss_neural_completion

    #para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="deco")
    #para_decode = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deco")
    #loss += weight_decay * tf.nn.l2_loss(para_decode)
    
    #para_encode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="enco_second")
    para_encode = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="enco_second")
    loss += weight_decay*tf.nn.l2_loss(para_encode[0])
    loss += weight_decay*tf.nn.l2_loss(para_encode[1])
    
    #accuracy = masked_accuracy(logits, lbl_in, msk_in, neg_msk)
    accuracy = calculate_accuracy(logits, lbl_in)

    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    #saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    print("Start to train")
    with tf.Session() as sess:
        sess.run(init_op)    

        train_loss_avg = 0
        train_acc_avg = 0
        
        #neg_mask, label_neg = generate_mask(labels, len(train_arr))
        neg_mask = mask_neg
        label_neg = label_neg
        
        for epoch in range(nb_epochs):
            
            t=time.time()
            
            tr_step = 0
           
            tr_size = fea_list[0].shape[0]  
            
            
            # ================   training    ============
            while tr_step * batch_size < tr_size:   

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}       
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}   
                fd4 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(inter_in_list, interaction_list)}   
                fd3 = {lbl_in: y_train,   
                       msk_in: train_mask,       
                       neg_msk: neg_mask,
                       is_train: True,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}
                fd = fd1
                fd.update(fd2)
                fd.update(fd4)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train, embed = sess.run([train_op, loss, accuracy, att_val, embed_list],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1
                
            print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,acc_tr, time.time()-t))    
            
            
        print("Finish traing.")
        
        # ================   test    ========================================================================================================
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd4 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(inter_in_list, interaction_list)}
            fd3 = {lbl_in: y_test,
                   msk_in: test_mask,       
                   neg_msk: neg_mask,
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}
            fd = fd1
            fd.update(fd2)
            fd.update(fd4)
            fd.update(fd3)
            out_come, loss_value_ts, acc_ts, jhy_final_embedding, embed = sess.run([logits, loss, accuracy, final_embedding, embed_list],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        out_come = out_come.reshape((1373,173))
        test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((1373,173)))
        test_labels, scores = ROC(out_come,labels, test_arr,test_negative_samples)
        sess.close()
        return test_labels, scores

        #sess.close()
