import sys
sys.path.append('../')
import models as m


def get_fair_Ks(Ks_tt, M, even=False):
    """ Get fair parameters.
    Wrapper function that calls compute_fair multiple times.
    Ks_tt is supposed to be an integer or array of integers of K values
    for the TT model. M is the dimension of the data.
    """
    Ks_cp = []
    Ks_gmm = []
    for K in Ks_tt:
        K_gmm, K_cp = compute_fair(K, M, even)
        Ks_cp.append(K_cp)
        Ks_gmm.append(K_gmm)
    return Ks_cp, Ks_gmm

def get_free_params(Ks_tt,M):
    """ Given a list of Tensor Train K computes the corresponding
    number of free parameters
    """
    free_params = []
    for K in Ks_tt:
        n_tt = m.TensorTrainGaussian(K, M).n_parameters()
        free_params.append(n_tt)
    
    return free_params


def compute_fair(K_tt, M, even=False):
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
    n_gmm = m.GMM(K_gmm, M).n_parameters()
    while n_gmm < n_tt:
        K_gmm += addi
        n_gmm = m.GMM(K_gmm, M).n_parameters()
    # n_gmm = K_gmm * (1 + M + M*M)
    # while n_gmm < n_tt:
    #     K_gmm += addi
    #     n_gmm = K_gmm * (1 + M + M*M)

    # number of free parameters for CP
    K_cp = K_gmm
    n_cp = m.CPGaussian(K_cp, M).n_parameters()
    while n_cp < n_tt:
        K_cp += addi
        n_cp = m.CPGaussian(K_cp, M).n_parameters()

    return K_gmm, K_cp
