import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import time
tfd = tfp.distributions
tfm = tf.math
import data as d
import models as m
from tqdm import tqdm

#%% Load data
N = 10000
data_names = d.get_toy_names()
name = data_names[7]

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
K = 12 # Number of components
M = 2 # Dimension of data
model = m.TensorTrainGaussian(K, M,seed = 2)

EPOCHS = 300
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%% Train model 
# Fit the model
losses = []
start_time = time.time()
for epoch in tqdm(range(EPOCHS),desc='Training TT'):    
    loss = 0
    for i,x in enumerate(dataset):
        loss += model.train_step(x,optimizer) 
    losses.append(loss.numpy()/len(dataset))
        
end_time = time.time()
print(f'Training time elapsed: {int(end_time-start_time)} seconds')
print(f'Final loss: {losses[-1]}')

f,ax = plt.subplots()
ax.plot(range(len(losses)),np.array(losses))
ax.set_title('Training loss')
ax.set_xlabel('iteration')
plt.show()

#%% Plot result

f,ax = plt.subplots(figsize=(8,8))
utl.plot_contours(ax, data, model,alpha=0.1)
ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_contour.png',dpi=300)


f,ax = plt.subplots(figsize=(8,5))
utl.plot_density(ax, model,cmap='hot')
ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_density.png',dpi=300)

integrand = utl.unitTest(model,limits=[-6,6])
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')


