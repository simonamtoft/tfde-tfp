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
tfd = tfp.distributions
tfm = tf.math

#%% Load data
N = 2000
data_names = d.get_toy_names()
name = data_names[0]

data = d.get_ffjord_data(name,batch_size=N)

# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(name + f' with {N} points')
plt.show()

# Split into batches
batch_size = 100
dataset = d.to_tf_dataset(data, batch_size=batch_size)

#%% Define model and training parameters
K = 5 # Number of components
M = 2 # Dimension of data

model = m.TensorRingGaussian(K,M)

EPOCHS = 20
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%% Train model 
losses = model.fit(dataset,EPOCHS,optimizer)

f,ax = plt.subplots()
ax.plot(range(len(losses)),np.array(losses))
ax.set_title('Training loss')
ax.set_xlabel('iteration')
plt.show()

#%%

integrand = utl.unitTest(model,limits=[-6,6])
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')

# f,ax = plt.subplots(figsize=(8,8))
# utl.plot_contours(ax, data, model,alpha=0.1)
# ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
# plt.show()



















