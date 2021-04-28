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

# name = 'toy'; N = 10000
# data = d.get_ffjord_data('checkerboard',batch_size=N)

# X_train = data[:int(N*0.8)]
# X_val = data[int(N*0.8):int(N*0.9)]
# X_test = data[int(N*0.9):]

# M = X_train.shape[1]

# print(f'\nX_train.shape = {X_train.shape}')
# print(f'\nX_val.shape = {X_val.shape}')
# print(f'\nX_train.shape = {X_train.shape}')
# print(name + ' data loaded...')
#%% Select number of components

# Ks_tt = [3,4,5]
# Ks_cp, Ks_gmm = utl.get_fair_Ks(Ks_tt, M)
# free_params = utl.get_free_params(Ks_tt,M)

#%% Create small sub-sets
# # Train on small subset
# N_small = 1000
# batch_size = 200
# idx_train = np.random.choice(np.arange(X_train.shape[0]),size=N_small)
# X_train_small = X_train[idx_train]

# # Split into batches
# ds_train = d.to_tf_dataset(X_train, batch_size=batch_size)
# ds_train_small = d.to_tf_dataset(X_train_small, batch_size=batch_size)

# # Other datasets
# ds_val = d.to_tf_dataset(X_val, batch_size=batch_size)
# ds_test = d.to_tf_dataset(X_test, batch_size=batch_size)

#%% Train models
# epochs = 100
# N_init = 1
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# N_TIMES = 3

# # Initialize arrays
# losses_train = np.zeros((N_TIMES,3,len(Ks_tt)))
# losses_test = np.zeros((N_TIMES,3,len(Ks_tt)))
# test_size = X_test.shape[0]

# for k in range(N_TIMES):
#     print(f'\nIter {k+1}/{N_TIMES}')
#     for i,K_tt in tqdm(enumerate(Ks_tt),'Training TT',total=len(Ks_tt),
#                        position=0,leave=True):
#         model = m.TensorTrainGaussian(K_tt, M)
        
#         loss_TT = model.fit(ds_train_small, epochs,optimizer, mute=True,
#                 N_init = N_init,earlyStop=False)
#         losses_train[k,0,i] = loss_TT[-1]
        
#         # Iterate over validation set
#         loss_test = np.zeros(test_size,dtype=np.float32)
#         for j,x in enumerate(ds_test):
#             loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
#         losses_test[k,0,i] = -np.mean(loss_test)
        
#     for i,K_cp in tqdm(enumerate(Ks_cp),'Training CP',total=len(Ks_tt),
#                        position=0,leave=True):
#         model = m.CPGaussian(K_cp, M)
        
#         loss_CP = model.fit(ds_train_small,epochs,optimizer,mu_init='random',
#                 mute=True,N_init=N_init,earlyStop=False)
#         losses_train[k,1,i] = loss_CP[-1]
        
#         # Iterate over validation set
#         loss_test = np.zeros(test_size,dtype=np.float32)
#         for j,x in enumerate(ds_test):
#             loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
#         losses_test[k,1,i] = -np.mean(loss_test)
    
#     for i,K_gmm in tqdm(enumerate(Ks_gmm),desc='Training GMM',total=len(Ks_gmm)):
#         model = m.GMM(K_gmm, M)
#         loss_GMM = model.fit_full(X_train_small, epochs, mu_init='random',N_init=N_init)
#         losses_train[k,2,i] = loss_GMM
        
#         # Iterate over validation set
#         loss_test = np.zeros(test_size,dtype=np.float32)
#         for j,x in enumerate(ds_test):
#             loss_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()  
#         losses_test[k,2,i] = -np.mean(loss_test)
    
    
#%% Plot results
# std = np.std(losses_test,axis=0)
# mean = np.mean(losses_test,axis=0)

# Results from run on google colab:
mean1 = np.array([[31.28134473, 26.18926748, 25.79159737, 25.69056829, 25.81786919,
        26.00662549, 26.37001165, 27.2899704 , 27.52616946],
       [28.82295354, 27.12733014, 27.02854029, 27.07364337, 27.23634275,
        27.69497236, 28.28850873, 28.87738037, 29.33570671],
       [28.06180445, 26.08420372, 26.23148727, 26.50916036, 27.00097338,
        27.40521685, 29.15374056, 31.98993429, 34.57981873]])
std1 = np.array([[1.56607450e+00, 1.89024815e-01, 8.48938645e-02, 1.37892231e-01,
        6.43582850e-02, 2.04786407e-02, 3.41130578e-02, 1.80851992e-01,
        9.42829618e-02],
       [4.97195986e-01, 1.07639885e-01, 4.31034188e-02, 8.87446987e-02,
        6.91173265e-02, 1.13135864e-01, 2.69755455e-01, 4.12644235e-01,
        1.33456409e-01],
       [2.37888170e-06, 1.03075281e-02, 1.72597964e-02, 4.83839153e-02,
        1.09413603e-01, 1.43400299e-01, 9.56495082e-02, 5.00015593e-02,
        2.97501331e-01]])
free_params = [64, 570, 1012, 1580, 3094, 4040, 6310, 9084, 10660]

f,ax = plt.subplots(figsize=(8,5))
for i in range(3):
    ax.errorbar(free_params, mean1[i],std1[i])
ax.legend(['TT','CP','GMM'])
ax.set_title('Learning rate for hepmass data with subset size = 1000')
ax.set_xlabel('Free parameters')
ax.set_ylabel('Negative log-likelihood')
ax.grid('on')
plt.show()
f.savefig('../figures/expressibility_test/hepmass_subsample_1000.png',dpi=300)


# Results from run on google colab:
mean2 = np.array([[29.95897547, 25.76902962, 25.08886782, 24.82613182, 24.38115501,
        24.22538503, 24.04525693, 23.91431808, 23.90615845],
       [28.29706001, 26.62859917, 26.29527219, 26.04671478, 25.7663784 ,
        25.66430982, 25.47564379, 25.50666936, 25.38289007],
       [27.94120471, 25.61610031, 25.46621831, 25.29117203, 25.12056033,
        24.92062314, 24.7362512 , 24.76343791, 24.67150243]])
std2 = np.array([[1.07947180e-01, 1.20236660e-01, 4.38255973e-02, 4.54991182e-02,
        5.68246632e-02, 9.59704649e-02, 4.75701141e-02, 4.01236521e-02,
        3.48672846e-02],
       [2.62348974e-01, 2.49679374e-01, 1.28659985e-01, 1.66927416e-01,
        4.03232198e-02, 5.63852171e-02, 6.59607050e-02, 7.11474680e-02,
        5.82767896e-02],
       [2.37888170e-06, 2.69739830e-06, 3.71103564e-02, 4.71366810e-03,
        5.66818597e-02, 7.09506095e-03, 1.58806716e-02, 5.25517798e-02,
        1.35955739e-01]])
f,ax = plt.subplots(figsize=(8,5))
for i in range(3):
    ax.errorbar(free_params, mean2[i],std2[i])
ax.legend(['TT','CP','GMM'])
ax.set_title('Learning rate for hepmass data with subset size = 10000')
ax.set_xlabel('Free parameters')
ax.set_ylabel('Negative log-likelihood')
ax.grid('on')
plt.show()
f.savefig('../figures/expressibility_test/hepmass_subsample_10000.png',dpi=300)


f,ax = plt.subplots(1,2,figsize=(12,5),sharey=True)
for i in range(3):
    ax[0].errorbar(free_params, mean1[i],std1[i])
for i in range(3):
    ax[1].errorbar(free_params, mean2[i],std2[i])
ax[0].legend(['TT','CP','GMM'])
ax[1].legend(['TT','CP','GMM'])
ax[0].set_title('Hepmass data with subset size = 1000')
ax[1].set_title('Hepmass data with subset size = 10000')
ax[0].set_xlabel('Free parameters')
ax[0].set_ylabel('Negative log-likelihood')
ax[0].grid('on')
ax[1].set_xlabel('Free parameters')
ax[1].set_ylabel('Negative log-likelihood')
ax[1].grid('on')
plt.show()
f.savefig('../figures/expressibility_test/hepmass_subsample.png',dpi=300)


