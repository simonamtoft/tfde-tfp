import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import models as m
import datasets as d
tfd = tfp.distributions
tfm = tf.math

#%% Load data
dataset_names = d.get_dataset_names()
name = dataset_names[1]
data, X_train, X_val, X_test = d.load_data(name)
# X_train = d.get_ffjord_data('checkerboard',batch_size=1000)

# Dimension of data
M = X_train.shape[1]

# Convert to float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# Split into batches
batch_size = 200
dataset = d.to_tf_dataset(X_train, batch_size=batch_size)

# print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(name)
print('Data loaded...')


#%% Define model and training parameters
K = 10 # Number of components
# model = m.CPGaussian(K,M)
model = m.TensorTrainGaussian(K,M)

EPOCHS = 100
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# losses = model.fit(dataset,EPOCHS,optimizer,N_init=1)


#%%
# epochs = 100
# final_loss = 2.164144319481929
# time = 1171 s
# plt.plot(losses)
# plt.show()
#%%
# for i, x in enumerate(dataset):
#     break
# def logdot(a, b):
#     # max_a, max_b = np.max(a), np.max(b)
#     max_a = tfm.reduce_max(a)
#     max_b = tfm.reduce_max(b)
#     exp_a, exp_b = a - max_a, b - max_b
#     exp_a = tfm.exp(exp_a)
#     exp_b = tfm.exp(exp_b)
#     c = exp_a @ exp_b
#     c = tfm.log(c)
#     c += max_a + max_b
#     return c

# def logdotexp(A, B):
#     max_A = tfm.reduce_max(A,axis=2,keepdims=True)
#     max_B = tfm.reduce_max(B,axis=1,keepdims=True)
#     C = tfm.exp(A - max_A) @ tfm.exp(B-max_B)
#     C = tfm.log(C)
#     C += max_A + max_B
#     return C
# def log_space_product_tf(A, B):
#     Astack = tf.transpose(tf.stack([A]*A.shape[1],axis=1),perm=[0,3,2,1])
#     Bstack = tf.transpose(tf.stack([B]*B.shape[2],axis=1),perm=[0,2,1,3])
#     C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
#     return C
# def log_space_product_vector_tf(a, B):
#     Astack = tf.transpose(a,perm=[0,2,1])
#     Bstack = tf.transpose(B,perm=[0,1,2])
#     C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
#     return C

# # Go from logits -> weights
# wk0 = tf.nn.softmax(model.wk0_logits, axis=1) # axis 1 as it is (1, K0)
# W = [tf.nn.softmax(model.W_logits[i], axis=0) for i in range(model.M)]
# sigma = tfm.softplus(model.pre_sigma)
  

# # exp
# product = tf.eye(wk0.shape[1])
# for i in range(model.M):
#   result = tfm.exp(
#       tfm.log(W[i]) + tfd.Normal(model.mu, sigma).log_prob(
#           x[:, tf.newaxis, tf.newaxis, i]
#       )
#   )
#   product = product @ tf.transpose(result, perm=[0, 2, 1])
#   # print(tfm.log(product))
# likelihoods = tf.squeeze(tf.reduce_sum(tf.squeeze(wk0 @ product, axis=1), axis=1))
# log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
# print(log_likelihoods)

# # log
# print('\n')
# product = tfm.log(W[0]) + tfd.Normal(model.mu, sigma).log_prob(
#           x[:, tf.newaxis, tf.newaxis, 0])
# product = tf.transpose(product, perm=[0, 2, 1])
# # print(product)
# for i in range(1,model.M):
#     result = tfm.log(W[i]) + tfd.Normal(model.mu, sigma).log_prob(
#           x[:, tf.newaxis, tf.newaxis, i])    
#     product = log_space_product_tf(product, tf.transpose(result, perm=[0, 2, 1]))
#     # print(product)
# prod = log_space_product_vector_tf(tf.expand_dims(tfm.log(wk0),axis=1),product)
# log_likelihoods = tf.reduce_logsumexp(prod,axis=1)
# print(log_likelihoods)

#%%%
# from scipy.special import logsumexp
# def log_space_product(A, B):
#     Astack = np.stack([A]*A.shape[0]).transpose(2,1,0)
#     Bstack = np.stack([B]*B.shape[1]).transpose(1,0,2)
#     return logsumexp(Astack+Bstack, axis=0)
# def log_space_product(A, B):
#     Astack = np.stack([A]*A.shape[1],axis=1).transpose(0,3,2,1)
#     Bstack = np.stack([B]*B.shape[2],axis=1).transpose(0,2,1,3)
#     return logsumexp(Astack+Bstack, axis=1)
# def log_space_product_vector(a, B):
#     Astack = np.transpose(a,axes=(0,2,1))
#     Bstack = np.transpose(B,axes=(0,1,2))
#     C = logsumexp(Astack+Bstack, axis=1)
#     return C

# def log_space_product_tf(A, B):
#     Astack = tf.transpose(tf.stack([A]*A.shape[1],axis=1),perm=[0,3,2,1])
#     Bstack = tf.transpose(tf.stack([B]*B.shape[2],axis=1),perm=[0,2,1,3])
#     C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
#     return C
# def log_space_product_vector_tf(a, B):
#     Astack = tf.transpose(a,perm=[0,2,1])
#     Bstack = tf.transpose(B,perm=[0,1,2])
#     C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
#     return C

# N = 10
# M = 5
# K = 4
# a = np.random.rand(N,M,K)
# b = np.random.rand(N,K,M)
# A = np.log(a)
# B = np.log(b)
# A_tf = tf.convert_to_tensor(A)
# B_tf = tf.convert_to_tensor(B)
# C_tf = tfm.exp(log_space_product_tf(A,B))
# C = a@b
# K = 2
# N = 1
# a = np.random.rand(N,K,K)
# w = np.random.rand(K,K)
# b = np.random.rand(N,K,K)
# f = np.random.rand(N,K,K)
# c = np.random.rand(1,K)

# A = np.log(a)
# B = np.log(b)
# C = np.log(c)
# W = np.log(w)
# F = np.log(f)

# A_tf = tf.convert_to_tensor(A)
# B_tf = tf.convert_to_tensor(B)
# F_tf = tf.convert_to_tensor(F)
# W_tf = tf.convert_to_tensor(W)
# C_tf = tf.convert_to_tensor(C)


# S = np.eye(K)
# prod = log_space_product(np.expand_dims(W,axis=0),A)
# S = log_space_product(np.expand_dims(S,axis=0),np.transpose(prod,axes=(0,2,1)))
# # S = log_space_product(A,np.transpose(B,axes=(0,2,1)))
# # S = log_space_product(S,np.transpose(F,axes=(0,2,1)))
# # S = log_space_product_vector(np.expand_dims(C,axis=1),S)

# S_tf = log_space_product_tf(tf.expand_dims(W_tf,axis=0),A_tf)
# # S_tf = log_space_product_tf(A_tf,tf.transpose(B_tf,perm=[0,2,1]))
# # S_tf = log_space_product_tf(S_tf,tf.transpose(F_tf,perm=[0,2,1]))
# # S_tf = log_space_product_vector_tf(tf.expand_dims(C,axis=1),S_tf)


# s = np.eye(K)
# prod = w@a
# s = s @ np.transpose(prod,axes=(0,2,1))
# # s = a@ np.transpose(b,axes=(0,2,1))
# # s = s @ np.transpose(f,axes=(0,2,1))
# # s = np.squeeze(c@ s)


# print(np.exp(S))
# print('\n')
# print(np.exp(S_tf.numpy()))
# print('\n')
# print(s)
