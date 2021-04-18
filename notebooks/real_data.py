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
data, X_train, X_val, X_test = d.load_data(name)
# X_train = d.get_ffjord_data('checkerboard',batch_size=1000)


# Split into batches
batch_size = 400
dataset = d.to_tf_dataset(X_train, batch_size=batch_size)

print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(name)
print('Data loaded...')


#%% Define model and training parameters
K = 10 # Number of components
M = X_train.shape[1] # Dimension of data
model = m.CPGaussian(K,M)
# model = m.TensorTrainGaussian(K,M)

EPOCHS = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# losses = model.fit(dataset,EPOCHS,optimizer,N_init=1)


#%%
N = 2000
data = d.get_ffjord_data('checkerboard',batch_size=N)

X_train = data[:int(N*0.8)]
X_val = data[int(N*0.8):int(N*0.9)]
X_test = data[int(N*0.9):]

# Parameters
Ks = [4,10,20]
model_name = 'TT'

CV_dict = utl.CV_holdout(X_train,X_val, Ks, model_name=model_name,
                         epochs=100, batch_size=400, N_init = 2)


# Extract information from dict
lr = CV_dict['learning_curves']
error_train = CV_dict['error_train']
error_val = CV_dict['error_val']

# Choose random index to show learning curve
idx = np.random.randint(0,len(Ks),1)[0]

f,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].plot(lr[idx],linewidth=2)
ax[0].set_title(f'Learning curve for K = {Ks[idx]}')
ax[0].set_xlabel('Iterations')
ax[0].set_ylabel('Negative log-likelihood')
ax[1].plot(Ks,error_train,'k.-',markersize=10)
ax[1].plot(Ks,error_val,'r.-',markersize=10)
ax[1].set_ylabel('Negative log-likelihood')
ax[1].set_xlabel('K')
ax[1].set_title('Selecting K for '+model_name+' model')
ax[1].legend(['Train','Validation'])
ax[1].grid('on')
plt.show()



