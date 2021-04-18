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


print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(name + ' data loaded...')


#%% Define model and training parameters
# K = 10 # Number of components
# M = X_train.shape[1] # Dimension of data
# # model = m.CPGaussian(K,M)
# model = m.TensorTrainGaussian(K,M)


# # Train on small subset
# idx = np.random.choice(np.arange(X_train.shape[0]),size=100000)
# X_train_small = X_train[idx]

# # Split into batches
# batch_size = 400
# dataset = d.to_tf_dataset(X_train_small, batch_size=batch_size)

# EPOCHS = 10
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# losses = model.fit(dataset,EPOCHS,optimizer,N_init=1)

#%%
# X_test_1 = X_test[:2010]

# batch_size = 400
# dataset_test = d.to_tf_dataset(X_test_1, batch_size=batch_size)

# y = np.zeros(X_test_1.shape[0],dtype=np.float32)
# for i,x in enumerate(dataset_test):
#     print(f'Indicies: {i*batch_size}:{i*batch_size + x.shape[0]}')
#     y[i*batch_size:i*batch_size+x.shape[0]] = model(x).numpy()
    

# a = model(tf.convert_to_tensor(X_test_1)).numpy()

#%% Holdout Cross-validation
# N = 2000
# data = d.get_ffjord_data('checkerboard',batch_size=N)

# X_train = data[:int(N*0.8)]
# X_val = data[int(N*0.8):int(N*0.9)]
# X_test = data[int(N*0.9):]

# Train on small subset
idx = np.random.choice(np.arange(X_train.shape[0]),size=100000)
X_train_small = X_train[idx]

# Parameters
# Ks = [4,10,20,50,100]
Ks = 100
model_name = 'TT'
N_init = 3
epochs = 1

CV_dict = utl.CV_holdout(X_train_small,X_val, Ks, model_name=model_name,
                          epochs=epochs, batch_size=400, N_init = N_init)


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



