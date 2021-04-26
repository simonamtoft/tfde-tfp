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

# name = 'toy'; N = 5000
# data = d.get_ffjord_data('checkerboard',batch_size=N)

# X_train = data[:int(N*0.8)]
# X_val = data[int(N*0.8):int(N*0.9)]
# X_test = data[int(N*0.9):]

# Train on small subset
idx = np.random.choice(np.arange(X_train.shape[0]),size=2000)
X_train_small = X_train[idx]

M = X_train.shape[1] # Dimension of data
print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(name + ' data loaded...')
#%% Holdout Cross-validation

Ks = [2,4,5]
model_name = 'TT'
epochs = 10
N_init = 2
batch_size = 400
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

CV_dict = utl.CV_holdout(X_train_small,X_val, Ks, model_name=model_name,
                         epochs=epochs, batch_size=batch_size, N_init = N_init)


# Extract information from dict
train_learning_curves = CV_dict['train_learning_curves']
val_learning_curves = CV_dict['val_learning_curves']
error_train = CV_dict['error_train']
error_val = CV_dict['error_val']

# Choose random index to show learning curve
idx = np.argmin(error_val)

f,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].plot(train_learning_curves[idx],'k-',linewidth=2)
ax[0].plot(val_learning_curves[idx],'r-',linewidth=2)
ax[0].set_title(f'Learning curve for K = {Ks[idx]}')
ax[0].set_xlabel('Iterations')
ax[0].set_ylabel('Negative log-likelihood')
ax[0].legend(['Train','Validation'])
ax[1].plot(Ks,error_train,'k.-',markersize=10)
ax[1].plot(Ks,error_val,'r.-',markersize=10)
ax[1].set_ylabel('Negative log-likelihood')
ax[1].set_xlabel('K')
ax[1].set_title('Selecting K for '+model_name+' model')
ax[1].legend(['Train','Validation'])
ax[1].grid('on')
plt.show()

#%% Train for optimal K
K_opt = Ks[idx]

if model_name == 'CP':
  model = m.CPGaussian(K_opt,M)
else:
  model = m.TensorTrainGaussian(K_opt,M)

# Split into batches
ds = d.to_tf_dataset(X_train, batch_size=batch_size)
ds_small = d.to_tf_dataset(X_train_small, batch_size=batch_size)

# losses = model.fit(ds_small,epochs,optimizer,N_init=N_init)

#%% Get test error

# ds_test = d.to_tf_dataset(X_test, batch_size=batch_size)
# errors_test = np.zeros(X_test.shape[0],dtype=np.float32)

# for j,x in enumerate(ds_test):
#   errors_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()

# test_loss = -tf.reduce_mean(errors_test).numpy()
# print(f'Test error : {test_loss}')

#%%
# def log_space_product_tf(A, B):
#     Astack = tf.transpose(tf.stack([A]*A.shape[1],axis=1),perm=[0,3,2,1])
#     Bstack = tf.transpose(tf.stack([B]*B.shape[2],axis=1),perm=[0,2,1,3])
#     C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
#     return C
# N = 1
# M = 2
# K = 2
# aaaaa = np.random.rand(N,M,K)
# bbbbb = np.random.rand(N,K,M)
# A = np.log(aaaaa)
# B = np.log(bbbbb)
# A = tf.convert_to_tensor(A)
# B = tf.convert_to_tensor(B)


# c = aaaaa @ bbbbb
# C = log_space_product_tf(A,B)

# print(np.log(c))
# print(C.numpy())

# import torch
# a = torch.tensor(aaaaa)
# b = torch.tensor(bbbbb)

# bsz, p, m = a.size()
# _,_,n = b.size()
# aa = a.unsqueeze(2).expand(bsz,p,n,m)
# bb = b.unsqueeze(1).transpose(2,3).expand(bsz,p,n,m)
# cc = torch.logsumexp(aa+bb,dim=1)


# print(cc.numpy())

# AA = tf.broadcast_to(tf.expand_dims(A, axis=2),[N,M,M,K])
# BB = tf.broadcast_to(tf.transpose(tf.expand_dims(B, axis=1),perm=[0,1,2,3]),[N,M,M,K])
# CC = tfm.reduce_logsumexp(AA+BB,axis=1)

# print(CC.numpy())


