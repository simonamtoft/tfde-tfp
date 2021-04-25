import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import datasets as d
import models as m
tfd = tfp.distributions
tfm = tf.math

#%% Load data
N = 2000
data_names = d.get_toy_names()
name = data_names[7]

data = d.get_ffjord_data(name,batch_size=N)

X_train = data[:int(N*0.8)]
X_val = data[int(N*0.8):int(N*0.9)]
X_test = data[int(N*0.9):]

# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(name + f' with {N} points')
plt.show()

# Split into batches
batch_size = 100
dataset_train = d.to_tf_dataset(X_train, batch_size=batch_size)
dataset_val = d.to_tf_dataset(X_val, batch_size=batch_size)

#%% Define model and training parameters
K = 10 # Number of components
M = 2 # Dimension of data
model = m.TensorTrainGaussian(K, M,seed = 2)

EPOCHS = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%% Train model 
losses_train, losses_val = model.fit_val(dataset_train, dataset_val,EPOCHS,optimizer)

f,ax = plt.subplots()
ax.plot(losses_train)
ax.plot(losses_val)
ax.set_title('Training loss')
ax.set_xlabel('iteration')
ax.legend(['Train','Validation'])
plt.show()


#%% Plot result

f,ax = plt.subplots(figsize=(8,8))
utl.plot_contours(ax, data, model,alpha=0.1,n_points=200)
ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_contour.png',dpi=300)


f,ax = plt.subplots(figsize=(8,5))
utl.plot_density(ax, model,cmap='hot',n_points=200)
ax.set_title(name+' with K = '+str(K)+', epochs = ' + str(EPOCHS))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_density.png',dpi=300)

integrand = utl.unitTest(model,limits=[-6,6],n_points=200)
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')






