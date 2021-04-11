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


def data_split(data, train_idx, test_idx, batch_size):
    # split into train and test data
    X_train = data[train_idx]
    X_test = data[test_idx]

    # Normalize to zero mean and unit variance
    mu = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    return X_train, X_test


def CV_1_fold(data, Ks=np.arange(4, 8, 2), model_name='TT', 
              CV_splits=5, epochs=200, optimizer=None, batch_size=100):
    """
    1-fold Cross validation function
    
    Ks          :   tuple or scalar. Should be a list of values that are divisible by 2
    model_name  :   Name of model to test with options 'TT', 'CP', or 'GMM'
    """ 

    if optimizer == None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if np.isscalar(Ks): # Transform
        Ks = (Ks,)
    
    M = data.shape[1] # Dimension of data
    
    # Split data and shuffle
    CV = KFold(n_splits=CV_splits, shuffle=True)
    
    # Initialize error arrays
    error_train = np.zeros((CV_splits, len(Ks)))
    error_test = np.zeros((CV_splits, len(Ks)))
    
    for i, (train_index, test_index) in enumerate(CV.split(data)):
        print(f'Cross-validation fold {i+1}/{CV_splits}')
        
        # split and normalize data
        X_train, X_test = data_split(data, train_index, test_index, batch_size)
        
        # create TF training dataset 
        ds_train = d.to_tf_dataset(X_train, batch_size=batch_size)
        
        for j, K in enumerate(Ks):
            # Fit model to training data
            if model_name == 'TT':
                model = m.TensorTrainGaussian(K, M)
                losses = model.fit(ds_train, epochs, optimizer, mute=True)
            elif model_name == 'CP':
                model = m.CPGaussian(K, M)
                losses = model.fit(ds_train, epochs, optimizer, mute=True, mu_init='random')
            elif model_name == 'GMM':
                model = GaussianMixture(n_components=K, covariance_type='full', n_init=5, init_params='random')
                model.fit(X_train)
                losses = [-model.score(X_train)]
            else:
                raise Exception('Provided model_name not valid')
            
            error_train[i, j] = losses[-1]
            
            # Get negative log-likelihood on test data
            if model_name == 'GMM': # Sklearn trains differently than us
                log_likelihoods_test = model.score_samples(X_test)
            else:
                log_likelihoods_test =  model(X_test)
            error_test[i, j] = -tf.reduce_mean(log_likelihoods_test).numpy()
    
        # Get average error across splits
        err_tr = np.mean(error_train, axis=0) # mean training error over the CV folds
        err_tst = np.mean(error_test, axis=0) # mean test error over the CV folds
    return err_tr, err_tst