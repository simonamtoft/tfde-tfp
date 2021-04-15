root = '../datasets/raw/'

from .power import POWER
from .gas import GAS
from .hepmass import HEPMASS
from .miniboone import MINIBOONE
from .bsds300 import BSDS300
from .mnist import MNIST
from .cifar10 import CIFAR10
from .datasets_selector import load_data
from .datasets_selector import get_dataset_names

# Toy data
from .toy_data import gen_cos, gen_line, gen_checkerboard, gen_2spirals, \
    gen_pinwheel, gen_8gaussians, gen_moons, gen_rings, gen_circles, \
    gen_swissroll, get_ffjord_data, get_toy_names, gen_8gaussians_d3split, \
    to_tf_dataset, split_data
