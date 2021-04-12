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

#%% Load data
dataset_names = d.get_dataset_names()
name = dataset_names[4]
data, X_train, X_val, X_test = d.load_data(name)

# Dimension of data
M = X_train.shape[1]

# Convert to float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# Split into batches
batch_size = 200
dataset = d.to_tf_dataset(X_train, batch_size=batch_size)

print('Data loaded...')


#%% Define model and training parameters
K = 8 # Number of components
# model = m.CPGaussian(K,M)
model = m.TensorTrainGaussian(K,M)

EPOCHS = 2
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%%




# #%% Train model 
# losses = model.fit(dataset,EPOCHS,optimizer,N_init = 1)

# f,ax = plt.subplots()
# ax.plot(range(len(losses)),np.array(losses))
# ax.set_title('Training loss')
# ax.set_xlabel('iteration')
# plt.show()