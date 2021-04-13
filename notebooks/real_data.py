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
name = dataset_names[0]
# data, X_train, X_val, X_test = d.load_data(name)
X_train = d.get_ffjord_data('checkerboard',batch_size=1000)

# Dimension of data
M = X_train.shape[1]

# Convert to float32
X_train = X_train.astype(np.float32)
# X_val = X_val.astype(np.float32)
# X_test = X_test.astype(np.float32)

# Split into batches
batch_size = 1
dataset = d.to_tf_dataset(X_train, batch_size=batch_size)

print('Data loaded...')


#%% Define model and training parameters
K = 2 # Number of components
# model = m.CPGaussian(K,M)
model = m.TensorTrainGaussian(K,M)

EPOCHS = 2
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# losses = model.fit(dataset,EPOCHS,optimizer,N_init=2)

#%%
for i, x in enumerate(dataset):
    break
def logdot(a, b):
    # max_a, max_b = np.max(a), np.max(b)
    max_a = tfm.reduce_max(a)
    max_b = tfm.reduce_max(b)
    exp_a, exp_b = a - max_a, b - max_b
    exp_a = tfm.exp(exp_a)
    exp_b = tfm.exp(exp_b)
    c = exp_a @ exp_b
    c = tfm.log(c)
    c += max_a + max_b
    return c

def logdotexp(A, B):
    max_A = tfm.reduce_max(A,axis=2,keepdims=True)
    max_B = tfm.reduce_max(B,axis=1,keepdims=True)
    C = tfm.exp(A - max_A) @ tfm.exp(B-max_B)
    C = tfm.log(C)
    C += max_A + max_B
    return C

# Go from logits -> weights
wk0 = tf.nn.softmax(model.wk0_logits, axis=1) # axis 1 as it is (1, K0)
W = [tf.nn.softmax(model.W_logits[i], axis=0) for i in range(model.M)]
sigma = tfm.softplus(model.pre_sigma)
  

# exp
product = tf.eye(wk0.shape[1])
for i in range(model.M):
  result = tfm.exp(
      tfm.log(W[i]) + tfd.Normal(model.mu, sigma).log_prob(
          x[:, tf.newaxis, tf.newaxis, i]
      )
  )
  product = product @ tf.transpose(result, perm=[0, 2, 1])
  print(tfm.log(product))
likelihoods = tf.squeeze(tf.reduce_sum(tf.squeeze(wk0 @ product, axis=1), axis=1))
log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
print(log_likelihoods)

# log
print('\n')
product = tfm.log(W[0]) + tfd.Normal(model.mu, sigma).log_prob(
          x[:, tf.newaxis, tf.newaxis, 0])
product = tf.transpose(product, perm=[0, 2, 1])
print(product)
for i in range(1,model.M):
    result = tfm.log(W[i]) + tfd.Normal(model.mu, sigma).log_prob(
          x[:, tf.newaxis, tf.newaxis, i])    
    product = logdot(product, tf.transpose(result, perm=[0, 2, 1]))
    print(product)
log_likelihoods = tf.reduce_logsumexp(tf.reduce_logsumexp(tfm.log(wk0)+product, axis=1),axis=1)
print(log_likelihoods)

# loglog
print('\n')
product = logdotexp(tf.expand_dims(tfm.log(W[0]),axis=0),tfd.Normal(model.mu, sigma).log_prob(
          x[:, tf.newaxis, tf.newaxis, 0]))
product = tf.transpose(product, perm=[0, 2, 1])
print(product)
for i in range(1,model.M):
    result = logdotexp(tf.expand_dims(tfm.log(W[i]),axis=0),tfd.Normal(model.mu, sigma).log_prob(
        x[:, tf.newaxis, tf.newaxis, i]))
    
    product = logdotexp(product, tf.transpose(result, perm=[0, 2, 1]))
    print(product)
log_likelihoods = tf.reduce_logsumexp(tf.reduce_logsumexp(tfm.log(wk0)+product, axis=1),axis=1)
print(log_likelihoods)

#%% Check gradients

# for i, x in enumerate(dataset):
#     break


# with tf.GradientTape() as tape:
#     log_likelihoods = model(x)
#     loss_value = -tf.reduce_mean(log_likelihoods)

# # Compute gradients
# tvars = model.trainable_variables
# gradients = tape.gradient(loss_value, tvars)
# optimizer.apply_gradients(zip(gradients, tvars))


#%% Example
# def logdot(a, b):
#     max_a, max_b = np.max(a), np.max(b)
#     exp_a, exp_b = a - max_a, b - max_b
#     np.exp(exp_a, out=exp_a)
#     np.exp(exp_b, out=exp_b)
#     c = np.dot(exp_a, exp_b)
#     np.log(c, out=c)
#     c += max_a + max_b
#     return c

# a = np.log(np.random.rand(3,4))
# b = np.log(np.random.rand(3,4))
# A = np.exp(a)
# B = np.exp(b)

# c = A @ B.T
# C = np.exp(logdot(a,b.T))




#%%

# #%% Train model 
# losses = model.fit(dataset,EPOCHS,optimizer,N_init = 1)

# f,ax = plt.subplots()
# ax.plot(range(len(losses)),np.array(losses))
# ax.set_title('Training loss')
# ax.set_xlabel('iteration')
# plt.show()