import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import utils as utl
import models as m
import datasets as d
tfd = tfp.distributions
tfm = tf.math

#%% Load data
# dataset_names = d.get_dataset_names()
# name = dataset_names[3]
# data, X_train, X_val, X_test = d.load_data(name)

name = 'toy'; N = 10000
data = d.get_ffjord_data('checkerboard',batch_size=N)

X_train = data[:int(N*0.8)]
X_val = data[int(N*0.8):int(N*0.9)]
X_test = data[int(N*0.9):]

M = X_train.shape[1]

print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(f'\nX_train.shape = {X_train.shape}')
print(name + ' data loaded...')
#%% Select number of components

Ks_tt = [3,5,10,15,20]
Ks_cp, Ks_gmm = utl.get_fair_Ks(Ks_tt, M)
free_params = utl.get_free_params(Ks_tt,M)

#%% Create small sub-sets
# Train on small subset
N_small = 1000
batch_size = 200
idx_train = np.random.choice(np.arange(X_train.shape[0]),size=N_small)
X_train_small = X_train[idx_train]

# Split into batches
ds_train = d.to_tf_dataset(X_train, batch_size=batch_size)
ds_train_small = d.to_tf_dataset(X_train_small, batch_size=batch_size)

# Other datasets
ds_val = d.to_tf_dataset(X_val, batch_size=batch_size)
ds_test = d.to_tf_dataset(X_test, batch_size=batch_size)

#%% Train models
epochs = 100
N_init = 1
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
N_TIMES = 3

# Initialize arrays
losses_train = np.zeros((N_TIMES,3,len(Ks_tt)))
losses_test = np.zeros((N_TIMES,3,len(Ks_tt)))
test_size = X_test.shape[0]

for k in range(N_TIMES):
    print(f'\nIter {k+1}/{N_TIMES}')
    for i,K_tt in tqdm(enumerate(Ks_tt),'Training TT',total=len(Ks_tt),
                       position=0,leave=True):
        model = m.TensorTrainGaussian(K_tt, M)
        
        loss_TT = model.fit(ds_train_small, epochs,optimizer, mute=True,
                N_init = N_init,earlyStop=False)
        losses_train[k,0,i] = loss_TT[-1]
        
        # Iterate over validation set
        loss_test = np.zeros(test_size,dtype=np.float32)
        for j,x in enumerate(ds_test):
            loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
        losses_test[k,0,i] = -np.mean(loss_test)
        
    for i,K_cp in tqdm(enumerate(Ks_cp),'Training CP',total=len(Ks_tt),
                       position=0,leave=True):
        model = m.CPGaussian(K_cp, M)
        
        loss_CP = model.fit(ds_train_small,epochs,optimizer,mu_init='random',
                mute=True,N_init=N_init,earlyStop=False)
        losses_train[k,1,i] = loss_CP[-1]
        
        # Iterate over validation set
        loss_test = np.zeros(test_size,dtype=np.float32)
        for j,x in enumerate(ds_test):
            loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
        losses_test[k,1,i] = -np.mean(loss_test)
    
    for i,K_gmm in tqdm(enumerate(Ks_gmm),desc='Training GMM',total=len(Ks_gmm)):
        model = m.GMM(K_gmm, M)
        loss_GMM = model.fit_full(X_train_small, epochs, mu_init='random',N_init=N_init)
        losses_train[k,2,i] = loss_GMM
        
        # Iterate over validation set
        loss_test = np.zeros(test_size,dtype=np.float32)
        for j,x in enumerate(ds_test):
            loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
        losses_test[k,2,i] = -np.mean(loss_test)
    
    
#%% Plot results

std = np.std(losses_test,axis=0)
mean = np.mean(losses_test,axis=0)

f,ax = plt.subplots(figsize=(8,5))
for i in range(3):
    ax.errorbar(free_params, mean[i],std[i])
ax.legend(['TT','CP','GMM'])
ax.set_title('Learning rate for ' + name + ' data')
ax.set_xlabel('Free parameters')
ax.set_ylabel('Negative log-likelihood')
ax.grid('on')
plt.show()
