import numpy as np
import sklearn
import pandas as pd


def get_real_data(name='POWER',path_to_data='real_data/'):
    """ Load the datasets used in the Ffjord paper """
    
    if name == 'POWER':
        data = load_POWER_data(path_to_data)
    elif name == 'MINIBOONE':
        data = load_MINIBOONE_data(path_to_data)
    else:
        raise Exception('Wrong data name')
        
    return data


def load_POWER_data(path):
    """ Loads the POWER dataset from:
        https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#
    """
    # load all data
    df = pd.read_csv(path+'household_power_consumption.txt', sep=';', header=0, low_memory=False)
    
    # Remove the two first columns 
    df = df.drop(['Date','Time'],axis=1)
    
    # Replace '?' with nans
    df.replace('?',np.nan,inplace=True)
        
    # Convert to numpy
    data = df.to_numpy().astype('float32')
    
    # Remove nan-observations
    nan_index = np.any(np.isnan(data),axis=1)
    data = data[~nan_index]
    
    # Normalize data
    data = normalize(data)
    return data

def load_MINIBOONE_data(path):
    """ Loads the MINIBOONE dataset from:
        https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
    """
    # Load data
    df = pd.read_csv(path +'MiniBooNE_PID.txt',header=None,skiprows=1,skipinitialspace=True,sep=' ')

    # Convert to numpy
    data = df.to_numpy().astype('float32')
    
    signal_events = 36499
    background_events = 93565
    
    # Normalize data
    data = normalize(data)
    return data


def normalize(X):
    """ Normalize to zero mean and unit variance """
    mu = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    
    X_normalized = (X-mu)/std

    return X_normalized







