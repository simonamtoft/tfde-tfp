import sys
sys.path.append('../')
import models as m


def get_fair_Ks(K_tt, M, even=False):
    """ Computes fair parameters.
    Computes parameters for the GMM and CP models, 
    such that the three models can be compared fairly.

    Input
      K_tt  (int)   : The value of K for the Tensor Train model.
      M     (int)   : The number of dimensions of the data, 
                      which the models are trained on
      even  (bool)  : Whether the Ks are required to be even.

    Return
      K_tt  (int) : Same as input.
      K_gmm (int) : The value of K for the GMM model.
      K_cp  (int) : The value of K for the CP model.
    """

    if even:
      addi = 2
    else:
      addi = 1

    # number of free parameters for TT
    n_tt = m.TensorTrainGaussian(K_tt, M).n_parameters()

    # number of free parameters for GMM
    K_gmm = K_tt
    n_gmm = K_gmm * (1 + M + M*M)
    while n_gmm < n_tt:
        K_gmm += addi
        n_gmm = K_gmm * (1 + M + M*M)

    # number of free parameters for CP
    K_cp = K_gmm
    n_cp = m.CPGaussian(K_cp, M).n_parameters()
    while n_cp < n_tt:
        K_cp += addi
        n_cp = m.CPGaussian(K_cp, M).n_parameters()

    return K_tt, K_gmm, K_cp