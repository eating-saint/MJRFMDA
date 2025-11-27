import scipy.io as sio
import numpy as np

intMat = sio.loadmat('../data/interaction.mat')
intMat = intMat['interaction']
print(intMat.shape,intMat)

P1 = sio.loadmat('../data/net1.mat')
P2 = sio.loadmat('../data/net2.mat')
P3 = sio.loadmat('../data/net3.mat')

F1 = np.loadtxt("../data/drug_features.txt")
F2 = np.loadtxt("../data/microbe_features.txt")

P1_v = P1['interaction']
P2_v = P2['net2']
P3_v = P3['net3_sub']
print(P1_v.shape, P1_v)
print(P2_v.shape, P2_v)
print(P3_v.shape, P3_v)

print(F1.shape)
print(F2.shape)

