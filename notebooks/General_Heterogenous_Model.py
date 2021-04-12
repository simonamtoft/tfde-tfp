#%%
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


K = 5
dists = [tfd.Normal, tfd.Normal]
params = [
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ]
]
modifiers = {
    0: {1: tfm.softplus},
    1: {1: tfm.softplus}
}

GTT = m.TensorTrainGeneral(K, dists, params, modifiers)
TT = m.TensorTrainGaussian(K, 2)
# %%

import datasets as d
data_names = d.get_toy_names()
print(data_names)

N = 10000
name = 'checkerboard'
data = d.get_ffjord_data(name, batch_size=N)

# Split into train, validation and test set
train, val, test = d.split_data(data)

# Split training set into batches
batch_size = 100
train_ds = d.to_tf_dataset(train, batch_size=batch_size)

EPOCHS = 500
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
losses = GTT.fit(train_ds, EPOCHS, optimizer)
losses2 = TT.fit(train_ds, EPOCHS, optimizer)

# %%
import utils as utl
# check integrand of density
integrand = utl.unitTest(GTT, limits=[-6, 6])
print(f'Density integrates to {round(integrand, 4)}')
print('It should be = 1.0')
# %%
f,ax = plt.subplots(1, 3, figsize=(16, 5))

# training loss
ax[0].plot(range(len(losses)), np.array(losses))
ax[0].set_title('Training loss')
ax[0].set_xlabel('iteration')

# contour plot
utl.plot_contours(ax[1], val, GTT, alpha=1)
ax[1].set_title(name + ' with K = '+str(K))

# density plot
utl.plot_density(ax[2], GTT)
ax[2].set_title('Density of ' + name + ' with K = ' + str(K))
plt.show()
# %%
f,ax = plt.subplots(1, 3, figsize=(16, 5))

# training loss
ax[0].plot(range(len(losses2)), np.array(losses2))
ax[0].set_title('Training loss')
ax[0].set_xlabel('iteration')

# contour plot
utl.plot_contours(ax[1], val, TT, alpha=1)
ax[1].set_title(name + ' with K = '+str(K))

# density plot
utl.plot_density(ax[2], TT)
ax[2].set_title('Density of ' + name + ' with K = ' + str(K))
plt.show()