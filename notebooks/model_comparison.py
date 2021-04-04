import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import time
import data as d
import models as m
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
tfd = tfp.distributions
tfm = tf.math

#%% Load data
N = 2000
data_names = d.get_toy_names()
name = data_names[7]

data = d.get_ffjord_data(name,batch_size=N)

# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(name + f' with {N} points')
plt.show()

# Split into train, validate and tes set
data_train, data_validate, data_test = d.split_data(data)

#%%
K_LINE = 5
M_DIM = 2

Ks = np.arange(2,40,1)
Ms = np.arange(2,3,1)
params = np.zeros((3,len(Ms),len(Ks)))
legends = []


for i,M in enumerate(Ms):
    for j,K in enumerate(Ks):
        model_Train = m.TensorTrainGaussian(K, M)
        model_CP = m.CPGaussian(K, M)
        
        
        params[0,i,j] = model_Train.n_parameters()
        params[1,i,j] = model_CP.n_parameters()
        params[2,i,j] = K+M*K+K*M*M
    legends.append('TT, M='+str(M))
    legends.append('CP, M='+str(M))
    legends.append('GMM, M='+str(M))
    
# Make a vertical line
val = params[0,Ms==M_DIM,Ks==K_LINE][0]
line = val*np.ones(Ks.shape)
legends.append('Equality line')
    
f,ax = plt.subplots(figsize=(10,5))
for i in range(len(Ms)):
    ax.plot(Ks,params[:,i].T,linewidth=3)
ax.plot(Ks,line,'--k',linewidth=3)
ax.legend(legends)
ax.set_xlabel('K')
ax.set_ylabel('Parameters')
ax.grid('on')
ax.set_title('Number of parameters')
ax.set_ylim([0,700])
plt.show()

# Find indicies that achieve the same number of trainable parameters
idx_TT = np.argmax(params[0,Ms==M_DIM] >= val)
idx_CP = np.argmax(params[1,Ms==M_DIM] >= val)
idx_GMM = np.argmax(params[2,Ms==M_DIM] >= val)

# Number of components 
K_TT = Ks[idx_TT]
K_CP = Ks[idx_CP]
K_GMM = Ks[idx_GMM]

print('Selected components:')
print(f'TensorTrain : {K_TT}')
print(f'CP          : {K_CP}')
print(f'GMM(sklean) : {K_GMM}')


#%% Perform cross validation
# Define training parameters
epochs = 10
model_name = 'TT'
CV_splits = 5


# Ks = np.arange(2,16,2)
print('\nCross-validating TT:\n-------------------')
train_error_TT, test_error_TT = utl.CV_1_fold(data_train,[K_TT],'TT',CV_splits,epochs)
print('\nCross-validating CP:\n-------------------')
train_error_CP, test_error_CP = utl.CV_1_fold(data_train,[K_CP],'CP',CV_splits,epochs)
print('\nCross-validating GMM:\n-------------------')
train_error_GMM, test_error_GMM = utl.CV_1_fold(data_train,[K_GMM],'GMM',CV_splits,epochs)


print('    Training error         Test error')
print(f'TT:  {train_error_TT[0]}         {test_error_TT[0]}')
print(f'CP:  {train_error_CP[0]}         {test_error_CP[0]}')
print(f'GMM: {train_error_GMM[0]}         {test_error_GMM[0]}')

# train_error, test_error = utl.CV_1_fold(data_train,[K_CP],'CP',CV_splits,epochs)


# f,ax = plt.subplots()
# ax.plot(Ks,train_error)
# ax.plot(Ks,test_error)
# ax.set_xlabel('Components [K]')
# ax.legend(['Train error','Test error'])
# ax.set_title('1 level Cross Validation')
# plt.show()

