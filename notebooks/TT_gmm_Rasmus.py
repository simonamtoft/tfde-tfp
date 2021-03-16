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

#%% Set data parameters
N = 2000
data_names = d.fjjordDataNames()
name = data_names[7]

data = d.get_ffjordData(name,batch_size=N)

# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(name + f' with {N} points')
plt.show()



#%% Define model and training parameters
K = 2 # Number of components
model = m.TensorTrainGaussian2D(K)

EPOCHS = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#%% Train model 
# Fit the model
losses = []
start_time = time.time()
for epoch in tqdm(range(EPOCHS),desc='Training TT'):
    loss = model.train_step(data,optimizer)
    losses.append(loss.numpy())
    # if epoch % 100 == 0:
    #     print("{}/{} mean neg log likelihood: {}".format(epoch, EPOCHS, loss))
        
end_time = time.time()
print(f'Training time elapsed: {int(end_time-start_time)} seconds')
print(f'Final loss: {loss.numpy()}')

f,ax = plt.subplots()
ax.plot(range(len(losses)),np.array(losses))
ax.set_title('Training loss')
ax.set_xlabel('iteration')
plt.show()

#%% Plot result

f,ax = plt.subplots(figsize=(5,5))
utl.plot_contours(ax, data, model)
ax.set_title(name+' with K = '+str(K))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_contour.png',dpi=300)


f,ax = plt.subplots(figsize=(5,5))
utl.plot_density(ax, model)
ax.set_title('Density of '+name+' with K = '+str(K))
plt.show()
# f.savefig('../figures/TensorTrain/'+name+'_K_'+str(K)+'_density.png',dpi=300)

integrand = utl.unitTest(model,limits=[-6,6])
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')


