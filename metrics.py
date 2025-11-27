import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score


def custom_loss(target, input, alpha):
    # 生成正样本和负样本的索引
    one_index = tf.where(tf.equal(target, 1))
    zero_index = tf.where(tf.equal(target, 0))

    # 计算损失
    loss = tf.losses.mean_squared_error(labels=target, predictions=input, reduction=tf.losses.Reduction.NONE)
    loss_sum = tf.reduce_sum(loss, axis=-1)  # 计算每个样本的损失

    # 根据索引选择正样本和负样本的损失
    one_index_loss = tf.gather(loss_sum, one_index)
    zero_index_loss = tf.gather(loss_sum, zero_index)

    return (1 - alpha) * tf.reduce_sum(one_index_loss) + alpha * tf.reduce_sum(zero_index_loss)

def masked_accuracy(preds, labels, mask, negative_mask, alpha):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  #负样本的标签值为0，正样本的标签值为1

    mask = tf.cast(mask, dtype=tf.float32)
    negative_mask = tf.cast(negative_mask, dtype=tf.float32)

    mask += negative_mask
    #mask = tf.cast(mask, dtype=tf.float32)

    error *= mask  #*代表点乘 计算误差时需要将负样本的累加
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))

def ROC(outs, labels, test_arr, label_neg):
    scores=[]
    for i in range(len(test_arr)):
        l=test_arr[i]
        scores.append(outs[int(labels[l,0]-1),int(labels[l,1]-1)])
    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i,0]),int(label_neg[i,1])])
    test_labels=np.ones((len(test_arr),1))
    temp=np.zeros((label_neg.shape[0],1))
    test_labels1=np.vstack((test_labels,temp))
    test_labels1=np.array(test_labels1,dtype=np.bool).reshape([-1,1])
    return test_labels1,scores


def calculate_performance_metrics(test_labels1,scores,threshold=0.5):
    scores = np.array(scores)
    # 将分数转换为二分类的预测结果
    predictions = (scores > threshold).astype(int)

    # 计算召回率
    recall = recall_score(test_labels1, predictions)

    # 计算 AUC
    auc = roc_auc_score(test_labels1, scores)

    # 计算 AUPR
    aupr = average_precision_score(test_labels1, scores)

    # 计算 F1 值
    f1 = f1_score(test_labels1, predictions)

    return recall, auc, aupr, f1

def calculate_accuracy(logits, labels):
    predictions = tf.cast(tf.greater(logits, 0.5), tf.int32)
    correct_predictions = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
