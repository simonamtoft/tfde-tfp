import datasets
import numpy as np
root_data = '../data/'       # where the datasets are

def load_data(name,logit=False,dequantize=False,flip = False):
    """
    Loads the dataset. Has to be called before anything else.
    :param name: string, the dataset's name
    """
    
    assert isinstance(name, str), 'Name must be a string'
    # global data
    
    
    if name == 'mnist':
        data = datasets.MNIST(logit=logit, dequantize=dequantize)
    elif name == 'bsds300':
        data = datasets.BSDS300()
    elif name == 'cifar10':
        data = datasets.CIFAR10(logit=logit, flip=flip, dequantize=dequantize)
    elif name == 'power':
        data = datasets.POWER()
    elif name == 'gas':
        data = datasets.GAS()
    elif name == 'hepmass':
        data = datasets.HEPMASS()
    elif name == 'miniboone':
        data = datasets.MINIBOONE()
    else:
        raise Exception('Unknown dataset')

    # get data splits
    X_train = data.trn.x
    X_val = data.val.x
    X_test = data.tst.x
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    return data, X_train, X_val, X_test

def get_dataset_names():
    return [
        'power', 'gas', 'hepmass', 'miniboone', 'bsds300',
        'cifar10', 'mnist']