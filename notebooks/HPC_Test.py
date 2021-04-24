#### IMPORTS ####
import sys
sys.path.append("../")
import pandas as pd
from os.path import join
from collections import Counter
import models as m
import utils as utl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tqdm import tqdm
import datasets as d
import dill

tfn = tf.nn
tfd = tfp.distributions
tfm = tf.math

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#### LOAD ADULT DATASET ####
adult_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
adult_data[1] = pd.factorize(adult_data[1])[0]
adult_data[3] = pd.factorize(adult_data[3])[0]
adult_data[5] = pd.factorize(adult_data[5])[0]
adult_data[6] = pd.factorize(adult_data[6])[0]
adult_data[7] = pd.factorize(adult_data[7])[0]
adult_data[8] = pd.factorize(adult_data[8])[0]
adult_data[9] = pd.factorize(adult_data[9])[0]
adult_data[13] = pd.factorize(adult_data[13])[0]
adult_data[14] = pd.factorize(adult_data[14])[0]

#### SPLIT INTO 90% TRAIN, 10% VALIDATION ####
data = adult_data.to_numpy().astype("float32")
N_tot = data.shape[0]
train_data = data[:int(N_tot*0.9)]
val_data = data[int(N_tot*0.9):]


#### HYPERPARAMETERS ####
Ks = [1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 89]
N_TT = [0]*len(Ks)
Ks.reverse()
M = data.shape[1]
Ks_CP = []
dif_cats = np.sum(np.max(train_data[:,[1,3,5,6,7,8,13,14]], 0)+1)
for K in Ks:
  K_CP  = K**2
  target = int(K + M*K*K + 6*2*K*K + (K*K*dif_cats) + K*K)
  curr = int(K_CP + 6*2*K_CP + K_CP*dif_cats + K_CP)
  while  curr < target:
    K_CP += 1
    curr = int(K_CP + 6*2*K_CP + K_CP*dif_cats + K_CP)
  Ks_CP.append(K_CP)
N_CP = [0]*len(Ks)
N_repeats = 3
EPOCHS = 500
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

subset_size = 1000 #train_data.shape[0]
train_subset = train_data[np.random.choice(train_data.shape[0], size=subset_size, replace=False),:]
train_batched = d.to_tf_dataset(train_subset, batch_size=500)
val_batched = d.to_tf_dataset(val_data, batch_size=500)

dists = [
  tfd.Normal, #Age
  tfd.Categorical, #Workclass
  tfd.Normal, #Final Weight
  tfd.Categorical, #Education
  tfd.Normal, #Education-Num
  tfd.Categorical, #Marital-Status
  tfd.Categorical, #Occupation
  tfd.Categorical, #Relationship
  tfd.Categorical, #Race
  tfd.Bernoulli, #Sex
  tfd.Normal, #Capital-Gain
  tfd.Normal, #Capital-Loss
  tfd.Normal, #Hours per week
  tfd.Categorical, #Native-Country
  tfd.Categorical # Earns more than 50K?
]
modifiers = {
  0: {1: tfm.softplus},
  2: {1: tfm.softplus},
  4: {1: tfm.softplus},
  10: {1: tfm.softplus},
  11: {1: tfm.softplus},
  12: {1: tfm.softplus},
}

train_means = np.mean(train_data, 0)
train_std = np.std(train_data, 0)

losses = np.zeros((2, N_repeats, len(Ks), EPOCHS))
validations = np.zeros((2, N_repeats, len(Ks)))
print("\n")
print(f"Ks to train: {Ks}")
for i, K in enumerate(Ks):
  print(f"Current K = {K}")
  for j in range(N_repeats):
    print(f"Repeat nr. {j+1}/{N_repeats}")
    params_TT = [
      [np.random.uniform(train_means[0]-2*train_std[0], train_means[0]+2*train_std[0], (K, K)), np.random.uniform(0, train_std[0], (K, K))], # Normal: mu, sigma,
      [np.ones((K, K, int(np.max(data[:,1]))+1))], # Categorical: Logits
      [np.random.uniform(train_means[2]-2*train_std[2], train_means[2]+2*train_std[2], (K, K)), np.random.uniform(0, train_std[2], (K, K))], # Normal: mu, sigma,
      [np.ones((K, K, int(np.max(data[:,3]))+1))], # Categorical: Logits
      [np.random.uniform(train_means[4]-2*train_std[4], train_means[4]+2*train_std[4], (K, K)), np.random.uniform(0, train_std[4], (K, K))], # Normal: mu, sigma,
      [np.ones((K, K, int(np.max(data[:,5]))+1))], # Categorical: Logits
      [np.ones((K, K, int(np.max(data[:,6]))+1))], # Categorical: Logits
      [np.ones((K, K, int(np.max(data[:,7]))+1))], # Categorical: Logits
      [np.ones((K, K, int(np.max(data[:,8]))+1))], # Categorical: Logits,
      [np.zeros((K, K))], # Bernoulli: Logits
      [np.random.uniform(train_means[10]-2*train_std[10], train_means[10]+2*train_std[10], (K, K)), np.random.uniform(0, train_std[10], (K, K))], # Normal: mu, sigma,
      [np.random.uniform(train_means[11]-2*train_std[11], train_means[11]+2*train_std[11], (K, K)), np.random.uniform(0, train_std[11], (K, K))], # Normal: mu, sigma,
      [np.random.uniform(train_means[12]-2*train_std[12], train_means[12]+2*train_std[12], (K, K)), np.random.uniform(0, train_std[12], (K, K))], # Normal: mu, sigma,
      [np.ones((K, K, int(np.max(data[:,13]))+1))], # Categorical: Logits
      [np.ones((K, K, int(np.max(data[:,14]))+1))], # Categorical: Logits,
    ]

    params_CP = [
      [np.random.uniform(train_means[0]-2*train_std[0], train_means[0]+2*train_std[0], (Ks_CP[i],)), np.random.uniform(0, train_std[0], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.ones((Ks_CP[i], int(np.max(data[:,1]))+1))], # Categorical: Logits
      [np.random.uniform(train_means[2]-2*train_std[2], train_means[2]+2*train_std[2], (Ks_CP[i],)), np.random.uniform(0, train_std[2], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.ones((Ks_CP[i], int(np.max(data[:,3]))+1))], # Categorical: Logits
      [np.random.uniform(train_means[4]-2*train_std[4], train_means[4]+2*train_std[4], (Ks_CP[i],)), np.random.uniform(0, train_std[4], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.ones((Ks_CP[i], int(np.max(data[:,5]))+1))], # Categorical: Logits
      [np.ones((Ks_CP[i], int(np.max(data[:,6]))+1))], # Categorical: Logits
      [np.ones((Ks_CP[i], int(np.max(data[:,7]))+1))], # Categorical: Logits
      [np.ones((Ks_CP[i], int(np.max(data[:,8]))+1))], # Categorical: Logits,
      [np.zeros((Ks_CP[i],))], # Bernoulli: Logits
      [np.random.uniform(train_means[10]-2*train_std[10], train_means[10]+2*train_std[10], (Ks_CP[i],)), np.random.uniform(0, train_std[10], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.random.uniform(train_means[11]-2*train_std[11], train_means[11]+2*train_std[11], (Ks_CP[i],)), np.random.uniform(0, train_std[11], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.random.uniform(train_means[12]-2*train_std[12], train_means[12]+2*train_std[12], (Ks_CP[i],)), np.random.uniform(0, train_std[12], (Ks_CP[i], ))], # Normal: mu, sigma,
      [np.ones((Ks_CP[i], int(np.max(data[:,13]))+1))], # Categorical: Logits
      [np.ones((Ks_CP[i], int(np.max(data[:,14]))+1))], # Categorical: Logits,
    ]
    
    H_CP = m.CPGeneral(Ks_CP[i], dists, params_CP, modifiers)
    H_TT = m.TensorTrainGeneral(K, dists, params_TT, modifiers)

    N_TT[i] = H_TT.n_parameters()
    N_CP[i] = H_CP.n_parameters()

    losses[0, j, i, :] = H_TT.fit(train_batched, epochs=EPOCHS, optimizer=optimizer, mute=True, tolerance=0)
    losses[1, j, i, :] = H_CP.fit(train_batched, EPOCHS=EPOCHS, optimizer=optimizer, mute=True)

    H_TT_val_losses = []
    H_CP_val_losses = []
    for _,x in enumerate(val_batched):
      H_TT_val_losses = np.concatenate([H_TT_val_losses, H_TT(x).numpy()])
      H_CP_val_losses = np.concatenate([H_CP_val_losses, H_CP(x).numpy()])

    validations[0, j, i] = -np.mean(H_TT_val_losses)
    validations[1, j, i] = -np.mean(H_CP_val_losses)

    with open(f"./dills/models/adult_N{subset_size}_I{i}_R{j}_models.dill", "wb") as handle:
      dill.dump({
        "H_TT": H_TT,
        "H_CP": H_CP,
        "N_TT": N_TT[i],
        "N_CP": N_CP[i]
      }, handle, protocol=4)

with open(f"./dills/adult_N{subset_size}_data.dill", "wb") as handle:
  dill.dump({
    "losses": losses,
    "validations": validations,
    "N_TT": N_TT,
    "N_CP": N_CP
  }, handle, protocol=4)

