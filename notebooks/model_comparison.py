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
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
tfd = tfp.distributions
tfm = tf.math

#%% Load data
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

# Split into train, validate and tes set
data_train, data_validate, data_test = d.split_data(data)


#%% Perform cross validation
# Define training parameters
epochs = 10
model_name = 'TT'
CV_splits = 5


Ks = np.arange(2,16,2)
train_error, test_error = utl.CV_1_fold(data_train,Ks,model_name,CV_splits,epochs = 10)


f,ax = plt.subplots()
ax.plot(Ks,train_error)
ax.plot(Ks,test_error)
ax.set_xlabel('Components [K]')
ax.legend(['Train error','Test error'])
ax.set_title('1 level Cross Validation')
plt.show()

