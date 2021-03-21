import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tf.math


class TensorTrainModel(tf.keras.Model):
    def __init__(self, K, M, seed=None):
        super(TensorTrainModel, self).__init__()
        self.K = K
        self.M = M

        # Enable reproduction of results
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize model weights
        Wk0 = np.ones((1, self.K))      
        W_logits = np.ones((self.M, self.K, self.K))

        # Define as TensorFlow variables
        self.wk0_logits = tf.Variable(Wk0, name="Wk0_logits", dtype=tf.dtypes.float32)
        self.W_logits = tf.Variable(W_logits, name="W_logits", dtype=tf.dtypes.float32)

    @tf.function
    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
    
        # Compute gradients
        tvars = self.trainable_variables
        gradients = tape.gradient(loss_value, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        return loss_value


class TensorTrainGaussian(TensorTrainModel):
    def __init__(self, K, M, seed=None):
        """ M-dimensional Tensor Train with Gaussian Mixture Models 
        
        Inputs
            K       (int)   :   The number of gaussian mixtures the model consits of
            M       (int)   :   The number of dimensions of the data
            seed    (int)   :   Set to other than none in order to reproduce results
        """
        super(TensorTrainGaussian, self).__init__(K, M, seed)
        mu = np.random.uniform(-4, 4, (self.M, self.K, self.K))
        pre_sigma = np.random.uniform(0, 5, (self.M, self.K, self.K))
        self.mu = tf.Variable(mu, name="mu", dtype=tf.dtypes.float32)
        self.pre_sigma = tf.Variable(pre_sigma, name="sigma", dtype=tf.dtypes.float32)

    def call(self, X):
        """ Calculates the log-likelihood of datapoint(s) with M-dimensions
            In matrix-multiplication format like z . (A (x) B)^T . (C (x) D)^T ...
        
        Input
            X   :    Datapoint(s) in M-dimensions.
        
        Return
            log likelihoods of data
        """

        # Ensure dimension of data is the same as model
        if X.shape[1] != self.M:
            raise Exception('Dataset has wrong dimensions')
        X = tf.cast(tf.reshape(X, (-1, self.M)), tf.float32)
  
        # Go from logits -> weights
        wk0 = tf.nn.softmax(self.wk0_logits, axis=1) # axis 1 as it is (1, K0)
        W = [tf.nn.softmax(self.W_logits[i], axis=0) for i in range(self.M)]
        
        # Go from raw values -> strictly positive values (ReLU approx.)
        sigma = [tfm.softplus(self.pre_sigma[i]) for i in range(self.M)]
  
        product = tf.eye(wk0.shape[1]) # start out with identity matrix
        for i in range(self.M):
          result = tfm.exp(
              tfm.log(W[i]) + tfd.Normal(self.mu[i], sigma[i]).log_prob(
                  # Make data broadcastable into (n, km, kn)
                  X[:, tf.newaxis, tf.newaxis, i]
              )
          ) # intermediary calculation in log-domain -> exp after.
          # Keep batch dimension in place, transpose matrices
          product = product @ tf.transpose(result, perm=[0, 2, 1])
        
        # In order: Squeeze (n, 1, k_last) -> (n, k_last).
        # Reduce sum over k_last into (n, )
        # Squeeze result to (n, ) if n > 1 or () if n == 1
        likelihoods = tf.squeeze(tf.reduce_sum(tf.squeeze(wk0 @ product, axis=1), axis=1))
  
        # add small number to avoid nan
        log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
        return log_likelihoods

    def sample(self, N):
        # TO-DO
        # Simply sample from categorical distributions based on wk0 and W logits
        # And then sample from corresponding distributions.
        pass

