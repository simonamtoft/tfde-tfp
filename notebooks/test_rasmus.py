import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import time
tfd = tfp.distributions
tfm = tf.math
import data as d
import models as m
from tqdm import tqdm

#%% Data
N = 5000
data_names = d.fjjordDataNames()
name = data_names[0]
data = d.get_ffjordData(name,batch_size=N)

#%% Define model
K = 4 # Number of components
M = 2
model = m.TensorTrainGaussian2D(K,seed = 2)

model2 = m.TensorTrainGaussian(K,M,seed = 2)
#%% 2D test
K = 4 # Number of components
model = m.TensorTrainGaussian2D(K,seed = 2)

params = model.get_params()
mu = params['mu'].astype(np.float32)
sigma = params['sigma'].astype(np.float32)
Wk0= params['Wk0'].astype(np.float32)
Wk2k1 = params['Wk2k1'].astype(np.float32)
Wk1k0 = params['Wk1k0'].astype(np.float32)

X = data[:5,:] # X.shape = (5,2)

dist = []
for i in range(K):
    for j in range(K):
        dist.append(tfd.Normal(mu[i,j],sigma[i,j]))
joint = tfd.JointDistributionSequential(dist)

z = Wk0

d = []
[d.append(X[:,0]) for s in range(K**2)]
# [d.append(X[0]) for s in range(K**2)]

A = Wk1k0
B = np.reshape(joint.prob_parts(d),(K,K,-1))
res1 = np.multiply(A[:,:,np.newaxis],B)


d = []
[d.append(X[:,1]) for s in range(K**2)]
# [d.append(X[1]) for s in range(K**2)]

C = Wk2k1
D = np.reshape(joint.prob_parts(d),(K,K,-1))
res2 = np.multiply(C[:,:,np.newaxis],D)

results = res1.T @ res2.T

val = np.sum(z @ results,axis=1)

print(np.log(val))
print(model(X).numpy())
print(model2(X).numpy())

#%% M-Dimensional test
M = 5
K = 4

Wk0 = np.random.rand(K)
W = np.random.rand(M,K,K)

X = np.random.rand(10,M)

dist = []
for i in range(K):
    for j in range(K):
        dist.append(tfd.Normal(mu[i,j],sigma[i,j]))
joint = tfd.JointDistributionSequential(dist)

# Calculate probability
z = Wk0

for i in range(M):
    d = []
    [d.append(X[:,i]) for s in range(K**2)]
    
    A = W[i]
    B = np.reshape(joint.prob_parts(d),(K,K,-1))
    res = np.multiply(A[:,:,np.newaxis],B)
    if i == 0:
        a = res.T
    else:
        a = a @ res.T

val = np.sum(z @ a, axis = 1)
print(np.log(val))