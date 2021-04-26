import sys
sys.path.append('../../')
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

# Set root of where data is
d.root = '../../datasets/raw/'

#%% Load data
dataset_names = d.get_dataset_names()
name = dataset_names[5]
data, X_train, X_val, X_test = d.load_data(name)

M = X_train.shape[1]

print(f'\nX_train.shape = {X_train.shape}')
print(f'\nX_val.shape = {X_val.shape}')
print(f'\nX_train.shape = {X_train.shape}')
print(name + ' data loaded...')

#%% Parameters
model_name = 'TT'
epochs = 10
N_init = 5 # Number of random initializations to do
batch_size = 100
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Train on small subset
N_small = 2000
idx_train = np.random.choice(np.arange(X_train.shape[0]),size=N_small)
idx_val = np.random.choice(np.arange(X_val.shape[0]),size=N_small//4)
X_train_small = X_train[idx_train]
X_val_small = X_val[idx_val]

#%% Perform hold-out cross validation

# List of component sizes to go through
# Ks = [6,10,15,25,30]
Ks = [2,4,6,8,10]

CV_dict = utl.CV_holdout(X_train_small,X_val_small, Ks, model_name=model_name,
                         epochs=epochs, batch_size=batch_size, N_init = N_init)

#%% Plot results of Cross-validation
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
f.suptitle(name)
plt.show()

#%% Fit new model      (Set optimal K either directly of from cross-validation)
idx = np.argmin(error_val)
K_opt = Ks[idx]
# K_opt = 30

if model_name == 'CP':
  model = m.CPGaussian(K_opt,M)
else:
  model = m.TensorTrainGaussian(K_opt,M)

epochs = 10

# Split into batches
ds_train = d.to_tf_dataset(X_train, batch_size=batch_size)
ds_train_small = d.to_tf_dataset(X_train_small, batch_size=batch_size)
ds_val = d.to_tf_dataset(X_val, batch_size=batch_size)
ds_val_small = d.to_tf_dataset(X_val_small, batch_size=batch_size)

# Train and plot
losses_train, losses_val = model.fit_val(ds_train_small, ds_val_small,
                                         epochs,optimizer,N_init = N_init)

f,ax = plt.subplots(figsize=(12,5))
ax.plot(losses_train)
ax.plot(losses_val)
ax.set_title('Training loss')
ax.set_xlabel('iteration')
ax.legend(['Train','Validation'])
ax.grid('on')
plt.show()

#%% Get loss on test-set
dataset_test = d.to_tf_dataset(X_test, batch_size=batch_size)
errors_test = np.zeros(X_test.shape[0],dtype=np.float32)

for j,x in tqdm(enumerate(dataset_test),desc='Testing',position=0,leave=True,total=len(dataset_test)):
  errors_test[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()

test_loss = -np.mean(errors_test)
print(f'\nTest error : {test_loss}')