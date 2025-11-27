
from function import cross_validation
from function import generate_train_negtive_samples
from train import train
from metrics import calculate_performance_metrics
import numpy as np
import scipy.io as sio

def main():
   labels = np.loadtxt("../data/adj.txt")
   reorder = np.arange(2470)
   np.random.shuffle(reorder)

   #seeds = [2341, 1367, 22, 1812, 1659]
   seeds = [2341]
   
   intMat = sio.loadmat('../data/interaction.mat')
   intMat = intMat['interaction']
   cv, num = 0, 5
   cv_data = cross_validation(intMat, seeds, cv, num)

   all_recall = []
   all_auc = []
   all_aupr = []
   all_f1_score = []
   
   for seed in cv_data.keys():
       for W, test_data, test_label in cv_data[seed]:
           train_arr = []
           input_data = W*intMat
           [x_pos, y_pos] = np.where(input_data==1)
           train_arr = [j for i in range(len(x_pos)) for j in range(labels.shape[0]) if x_pos[i]==labels[j,0]-1 and y_pos[i]==labels[j,1]-1]
           test_arr = list(set(reorder).difference(set(train_arr)))
           mask_neg, label_neg = generate_train_negtive_samples(W, intMat)
           test_labels, scores = train(train_arr, test_arr, mask_neg, label_neg)
           recall, auc, aupr, f1 = calculate_performance_metrics(test_labels,scores,threshold=0.5)
           all_recall.append(recall)
           all_auc.append(auc)
           all_aupr.append(aupr)
           all_f1_score.append(f1)
           print("Recall:", recall)
           print("AUC:", auc)
           print("AUPR:", aupr)
           print("F1 Score:", f1)

   avg_recall = np.mean(all_recall)
   avg_auc = np.mean(all_auc)
   avg_aupr = np.mean(all_aupr)
   avg_f1_score = np.mean(all_f1_score)

   print("Average Recall:", avg_recall)
   print("Average AUC:", avg_auc)
   print("Average AUPR:", avg_aupr)
   print("Average F1 Score:", avg_f1_score)
if __name__ == "__main__":
    main()           
