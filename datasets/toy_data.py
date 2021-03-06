# functionality to generate toy data.
# taken from FFJORD toy data generation
# https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py 

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import tensorflow as tf

DEFAULT_SPLIT = (0.4, 0.30, 0.3)


def get_toy_names():
    return [
        '8gaussians', 'swissroll', 'circles', 'rings', 'moons',
        'pinwheel', '2spirals', 'checkerboard', 'line', 'cos',
    ]


def get_ffjord_data(name='8gaussians', batch_size=200, rng=None):
    """ Wrapper function to get FFJORD toy data"""
    
    if name == '8gaussians':
        data = gen_8gaussians(rng, batch_size)
    elif name == 'swissroll':
        data = gen_swissroll(rng, batch_size)
    elif name == 'circles':
        data = gen_circles(rng, batch_size)
    elif name == 'rings':
        data = gen_rings(rng, batch_size)
    elif name == 'moons':
        data = gen_moons(rng, batch_size)
    elif name == 'pinwheel':
        data = gen_pinwheel(rng, batch_size)
    elif name == '2spirals':
        data = gen_2spirals(rng, batch_size)
    elif name == 'checkerboard':
        data = gen_checkerboard(rng, batch_size)
    elif name == 'line':
        data = gen_line(rng, batch_size)
    elif name == 'cos':
        data = gen_cos(rng, batch_size)
    else:
        raise Exception('FFJORD data name is incorrect')
        
    # Convert to float32 data type
    data = data.astype(np.float32)
    return data


# Convert the data to a TF dataset 
# from this: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/ 
def to_tf_dataset(data, batch_size=500):
    tf_data = tf.data.Dataset.from_tensor_slices((data))
    tf_data = tf_data.shuffle(buffer_size=1024).batch(batch_size)
    return tf_data


def split_data(data, validate_split=0.1, test_split=0.1):
    """ Splits data into train, validation and test set 
    By default dataset is split into:
        Train    : 80%
        Validate : 10%
        Test     : 10%
    """
    N_test = int(test_split * data.shape[0])
    N_validate = int(validate_split * data.shape[0]) + N_test

    data_test = data[:N_test]
    data_validate = data[N_test:N_validate]
    data_train = data[N_validate:]

    return data_train, data_validate, data_test


def gen_swissroll(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
    
    data = sklearn.datasets.make_swiss_roll(
        n_samples=batch_size, 
        noise=1.0
    )[0]
    data = data.astype("float32")[:, [0, 2]]
    data /= 5
    return data


def gen_circles(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    data = sklearn.datasets.make_circles(
        n_samples=batch_size, 
        factor=.5, 
        noise=0.08
    )[0]
    data = data.astype("float32")
    data *= 3
    return data


def gen_rings(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    n_samples4 = n_samples3 = n_samples2 = batch_size // 4
    n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, we set endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    X = np.vstack([
        np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
        np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
    ]).T * 3.0
    X = util_shuffle(X, random_state=rng)

    # Add noise
    X = X + rng.normal(scale=0.08, size=X.shape)

    return X.astype("float32")


def gen_moons(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return data


def gen_8gaussians(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
    scale = 4.

    centers = [
        (1, 0), 
        (-1, 0), 
        (0, 1), 
        (0, -1), 
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for _ in range(batch_size):
        point = rng.randn(2) * 0.5
        idx = rng.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset


def gen_8gaussians_d3split(rng=None, batch_size=200, d3split=DEFAULT_SPLIT):
    if rng is None:
        rng = np.random.RandomState()

    if np.sum(d3split) != 1.0:
        d3split = DEFAULT_SPLIT
        print(f'Split did not add up to 1. Using split: {d3split}')
    elif len(d3split) != 3:
        d3split = DEFAULT_SPLIT
        print(f'Split did not have 3 values. Using split: {d3split}')
    
    scale = 4.

    centers_1 = [
        # (1, 0), 
        (-1, 0), 
        (0, 1), 
        (0, -1), 
        # (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)), 
        # (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers_2 = [
        (1, 0), 
        (-1, 0), 
        # (0, 1), 
        (0, -1), 
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        # (1. / np.sqrt(2), -1. / np.sqrt(2)), 
        # (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers_3 = [
        (1, 0), 
        # (-1, 0), 
        (0, 1), 
        # (0, -1), 
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
        # (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers_1 = [(scale * x, scale * y) for x, y in centers_1]
    centers_2 = [(scale * x, scale * y) for x, y in centers_2]
    centers_3 = [(scale * x, scale * y) for x, y in centers_3]

    dataset = []
    centersd3 = [centers_1, centers_2, centers_3]
    for j in range(3):
        for _ in range(int(batch_size*d3split[j])):
            point = rng.randn(3) * 0.5
            idx = rng.randint(len(centersd3[j]))
            center = centersd3[j][idx]
            point[0] += center[0]
            point[1] += center[1]
            point[2] = j
            dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset[:, 0:2] /= 1.414
    return dataset


def gen_pinwheel(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = batch_size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([
        np.cos(angles), 
        -np.sin(angles), 
        np.sin(angles), 
        np.cos(angles)
    ])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))


def gen_2spirals(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x


def gen_checkerboard(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


def gen_checkerboard_d3split(rng=None, batch_size=200):
    data = gen_checkerboard(batch_size=batch_size)
    label = np.zeros_like(data[:, 0])
    
    for i in range(label.shape[0]):
        x1 = data[i, 0]
        x2 = data[i, 1]
        if (x1 <= -2 and x2 <= -2 or 
            x1 <= 0 and x2 >= 2):
            label[i] = 1
        elif (x1 >= 2 and x2 >= 2 or 
            x1 >= -2 and x1 <= 0 and x2 >=-2 and x2 <= 0):
            label[i] = 2

    return np.transpose(np.stack((data[:, 0], data[:, 1], label)))


def gen_line(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    x = rng.rand(batch_size) * 5 - 2.5
    y = x
    return np.stack((x, y), 1)


def gen_cos(rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
 
    x = rng.rand(batch_size) * 5 - 2.5
    y = np.sin(x) * 2.5
    return np.stack((x, y), 1)
