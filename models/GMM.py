import tensorflow as tf
import tensorflow_probability as tfp
import time
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
tfd = tfp.distributions   
tfm = tf.math

class GMM:
    def __init__(self, K, M, seed = None):
        super(GMM, self).__init__()
        """A wrapper for sklearns GaussianMixture that behaves as our 
        other implemented models.

        Parameters
        ----------
        K : int > 0
        Amount of components
        M : int > 0
        Amount of dimensions pr. component
        """
        self.K = K
        self.M = M

        # Enable reproduction of results
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize model weights
        mu = np.zeros((self.K,self.M))
        sigma = np.ones((self.K,self.M))
        W_logits = np.ones((1, self.K))      
        
         # Define as TensorFlow variables
        self.mu = tf.Variable(mu, name="mu", dtype=tf.dtypes.float32)
        self.sigma = tf.Variable(sigma, name="sigma", dtype=tf.dtypes.float32)
        self.W_logits = tf.Variable(W_logits, name="W_logits", dtype=tf.dtypes.float32)
        self.gmm_model = GaussianMixture(self.K, covariance_type='full')
        
        return None
    
    def __call__(self,data):
        if data.shape[1] != self.M:
            raise Exception('Data has wrong dimensions')
            
        log_likelihoods = self.gmm_model.score_samples(data)
        
        return tf.convert_to_tensor(log_likelihoods)
        
    def fit(self, data, EPOCHS=200, mu_init='kmeans', mute=False):
        """
        """

        losses = np.zeros((EPOCHS,))
        start_time = time.time()
        for epoch in tqdm(range(EPOCHS), desc='Training GMM', disable=mute):
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")
                model = GaussianMixture(self.K, covariance_type='full',n_init=5, 
                                        tol=1e-5, reg_covar=1e-6, init_params=mu_init,
                                        random_state = 0, max_iter = 1+epoch)
                model.fit(data)

            losses[epoch] = -np.mean(model.score_samples(data))
            
        # Set final model
        self.gmm_model = model
        self.mu = tf.Variable(model.means_, name="mu", dtype=tf.dtypes.float32)
        self.sigma = tf.Variable(model.covariances_, name="sigma", dtype=tf.dtypes.float32)
        self.W_logits = tf.Variable(model.weights_, name="W_logits", dtype=tf.dtypes.float32)
            
        end_time = time.time()
        if not mute:
            print(f'Training time elapsed: {int(end_time-start_time)} seconds')
            print(f'Final loss: {losses[-1]}')

        return losses
            
            
            
            