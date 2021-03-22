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
data = d.get_real_data('HEPMASS',path_to_data='../data/real_data/')
N = data.shape[0]

f,ax = plt.subplots()
im = ax.imshow(data,extent=[0,100,0,1],aspect='auto')
f.colorbar(im,ax=ax)
ax.set_title('data')
plt.show()


# Split into batches
dataset = d.to_tf_dataset(data, batch_size=100)

#%% Define model and training parameters
K = 10 # Number of components
M = data.shape[1] # Dimension of data
model = m.TensorTrainGaussian(K, M,seed = 2)

EPOCHS = 3
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