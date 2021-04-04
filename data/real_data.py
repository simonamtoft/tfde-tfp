import numpy as np
import sklearn
import pandas as pd
import os
from collections import Counter
from os.path import join


def get_real_names():
    return [
        'POWER', 'MINIBOONE', 'HEPMASS'
    ]


def get_real_data(name='POWER', path_to_data='real_data/'):
    """ Load the datasets used in the FFJORD paper """

    if name == 'POWER':
        data_train = load_POWER_data(path_to_data)
    elif name == 'MINIBOONE':
        data_train, data_validate, data_test = load_MINIBOONE_data(path_to_data)
    elif name == 'HEPMASS':
        data_train, data_validate, data_test = load_HEPMASS_data(path_to_data)
    else:
        raise Exception('Wrong data name')
        
    return data_train


def load_POWER_data(path):
    """ Loads the POWER dataset from:
        https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#
    """
    filename = 'household_power_consumption.txt'

    # Check if data file exists
    if not os.path.exists(path + filename):
        print("Data file for POWER is not present.\nDownload from https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#")
        return -1

    # load all data
    df = pd.read_csv(path + filename, sep=';', header=0, low_memory=False)
    
    # Remove the two first columns 
    df = df.drop(['Date','Time'], axis=1)
    
    # Replace '?' with nans
    df.replace('?', np.nan, inplace=True)
        
    # Convert to numpy
    data = df.to_numpy().astype('float32')
    
    # Remove nan-observations
    nan_index = np.any(np.isnan(data), axis=1)
    data = data[~nan_index]
    
    # Normalize data
    data = normalize(data)
    return data


def load_MINIBOONE_data(path):
    """ Loads the MINIBOONE dataset from:
        https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
    """
    filename = 'MiniBooNE_PID.txt'
    
    # Check if data file exists
    if not os.path.exists(path + filename):
        raise Exception("Data file for MINIBOONE is not present.\nDownload from https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification")
    
    # Remove some features from data (to mimic FFJORD paper preprocessing)
    if not os.path.isfile(path + "miniboone_data.npy"):
        preprocess_miniboone(path)
        
    # Load data
    data = np.load(path + "miniboone_data.npy").astype(np.float32)
    
    # Split into train, validate and test
    N_test = int(0.1 * data.shape[0])
    N_validate = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]
    
    
    # Stack data and normalize
    data = np.vstack((data_train, data_validate))
    
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


def load_HEPMASS_data(path='real_data/'):
    """ Loads the HEPMASS dataset from:
        http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """
    filename_train = '1000_train.csv.gz'
    filename_test = '1000_test.csv.gz'

    # Check if data file exists
    if not os.path.exists(path + filename_train) or not os.path.exists(path + filename_test):
        raise Exception("Data file for HEPMASS is not present.\nDownload from http://archive.ics.uci.edu/ml/datasets/HEPMASS")

    # Remove some features from data (to mimic ffjord paper preprocessing)
    if not os.path.isfile(path+"hepmass_train.npy"):
        preprocess_hepmass(path)
    
    # Load data
    data_train = np.load(path+"hepmass_train.npy").astype(np.float32)
    data_test = np.load(path+"hepmass_test.npy").astype(np.float32)
    
    # Split test into train, test and validate
    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]
    
    return data_train, data_validate, data_test


def preprocess_miniboone(path='real_data/'):
    filename = 'MiniBooNE_PID.txt'
    # NOTE: To remember how the pre-processing was done.
    data = pd.read_csv(path + filename, names=[str(x) for x in range(50)], delim_whitespace=True)
    # print(data.head())
    
    # data = data.as_matrix()
    data = data.to_numpy()
    # Remove some random outliers
    indices = (data[:, 0] < -100)
    data = data[~indices]
    
    # Remove nan-observations
    nan_index = np.any(np.isnan(data), axis=1)
    data = data[~nan_index]
    
    i = 0
    # Remove any features that have too many re-occuring real values.
    features_to_remove = []
    for feature in data.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
    
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    
    # Save result
    np.save(path + "miniboone_data.npy", data)
    return None


def preprocess_hepmass(path = 'real_data/'):
    filename_train = "1000_train.csv.gz"
    filename_test = "1000_test.csv.gz"
    
    print('Pre-processing hepmass data')
    print('This will take a couple of minutes...')

    # Load data (this takes some time)
    data_train = pd.read_csv(filepath_or_buffer=join(path, filename_train), index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, filename_test), index_col=False)
    
    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)
    
    # Normalize data
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s
    
    # Convert to numpy
    data_train, data_test = data_train.to_numpy(), data_test.to_numpy()
    
    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]
    
    # Save result
    np.save(path + "hepmass_train.npy", data_train)
    np.save(path + "hepmass_test.npy", data_test)
    return None


def normalize(X):
    """ Normalize to zero mean and unit variance """
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    X_normalized = (X-mu)/std

    return X_normalized
