import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import time
import datasets as d
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
    """ 1-fold Cross Validation
    
    Input
        data        :   The data to fit and test on. The method will split this 
                        into a training and testing self itself.
        Ks          :   Array or int of K values for the model
        model_name  :   Name of model to test ('TT', 'CP', 'GMM')
        epochs      :   How many epochs to use for fitting of the model
        optimizer   :   A tf.keras.optimizers to use for fitting the model
        batch_size  :   The desired batch size for the training data
    
    Return
        err_tr      :   Error on the training set
        err_tst     :   Error on the testing set
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
                train_loss = model.fit(ds_train, epochs, optimizer, mute=True)
                test_loss = model(X_test)
            elif model_name == 'CP':
                model = m.CPGaussian(K, M)
                train_loss = model.fit(ds_train, epochs, optimizer, mute=True, mu_init='random')
                test_loss = model(X_test)
            elif model_name == 'GMM':
                model = GaussianMixture(n_components=K, covariance_type='full', n_init=5, init_params='random')
                model.fit(X_train)
                train_loss = [-model.score(X_train)]
                test_loss = model.score_samples(X_test)
            else:
                raise Exception('Provided model_name not valid')
            
            error_train[i, j] = train_loss[-1]
            error_test[i, j] = -tf.reduce_mean(test_loss).numpy()
    
        # Get average error across splits
        err_tr = np.mean(error_train, axis=0) # mean training error over the CV folds
        err_tst = np.mean(error_test, axis=0) # mean test error over the CV folds
    return err_tr, err_tst

def CV_holdout(X_train,X_val, Ks=np.arange(4, 8, 2), model_name='TT', 
              epochs=200, optimizer=None, batch_size=100, N_init = 5):
    """ Holdout Cross Validation to find optimal K
    
    Input
        data        :   The data to fit and test on. The method will split this 
                        into a training and testing self itself.
        Ks          :   Array or int of K values for the model
        model_name  :   Name of model to test ('TT', 'CP', 'GMM')
        epochs      :   How many epochs to use for fitting of the model
        optimizer   :   A tf.keras.optimizers to use for fitting the model
        batch_size  :   The desired batch size for the training data
        N_init      :   How many initalizations the model should do
    
    Return CV_dict with values
        error_train      :   Error on the training set
        error_val        :   Error on the testing set
        learning_curves  : Learning curves for all the K
    """ 

    if optimizer == None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if np.isscalar(Ks): # Transform
        Ks = (Ks,)
        
    mute = True

    M = X_train.shape[1] # Dimension of data
    
    # create TF training dataset 
    ds_train = d.to_tf_dataset(X_train, batch_size=batch_size)
    ds_val = d.to_tf_dataset(X_val, batch_size=batch_size)
    
    # Initialize error arrays
    error_train = np.zeros((len(Ks)))
    error_val = np.zeros((len(Ks)))
    train_learning_curves = []
    val_learning_curves = []
    
    for i,K in tqdm(enumerate(Ks),desc='Fitting for K',total=len(Ks),position=0,leave=True):
        # Fit model to training data
        if model_name == 'TT':
            model = m.TensorTrainGaussian(K, M)
            train_loss,val_loss = model.fit_val(ds_train,ds_val,epochs,
                                                 optimizer, mute=mute, N_init=N_init)
        elif model_name == 'CP':
            model = m.CPGaussian(K, M)
            train_loss,val_loss = model.fit_val(ds_train,ds_val,epochs,
                                                 optimizer, mute=mute, N_init=N_init)
        # elif model_name == 'GMM':
        #     model = m.GMM(K,M)
        #     train_loss = model.fit(X_train, EPOCHS=epochs, mu_init='random', mute=mute)
        #     for j,x in enumerate(ds_test):
        #         test_loss[j*batch_size:j*batch_size+x.shape[0]] = model(x).numpy()
        else:
            raise Exception('Provided model_name not valid')
        
        train_learning_curves.append(train_loss)
        val_learning_curves.append(val_loss)
        error_train[i] = train_loss[-1]
        error_val[i] = val_loss[-1]
        
    CV_dict = {
        'error_train' : error_train,
        'error_val' : error_val,
        'train_learning_curves' : train_learning_curves,
        'val_learning_curves' : val_learning_curves
        }

    return CV_dict