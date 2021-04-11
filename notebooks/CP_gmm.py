import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import data as d
import models as m
tfd = tfp.distributions
tfm = tf.math

#%% Set data parameters
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

# Split into batches
batch_size = 200
dataset = d.to_tf_dataset(data, batch_size=batch_size)

#%% Define model and training parameters
K = 8 # Number of components
M = data.shape[1] # Number of dimensions in data
# model = m.CPGaussian(K,M)
model = m.GMM(K,M)

EPOCHS = 200
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%% Train model 
# losses = model.fit(dataset,EPOCHS,optimizer,'kmeans')
losses = model.fit(data,10,'kmeans')

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
# f.savefig('../figures/CP/'+name+'_K_'+str(K)+'_contour.png',dpi=300)


f,ax = plt.subplots(figsize=(8,5))
utl.plot_density(ax, model,cmap='hot')
ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
plt.show()
# f.savefig('../figures/CP/'+name+'_K_'+str(K)+'_density.png',dpi=300)

integrand = utl.unitTest(model,limits=[-6,6])
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')
























