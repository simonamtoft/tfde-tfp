import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tf.math

class TensorTrainGaussian2D(tf.keras.Model):
    def __init__(self, K, M=None):
        super(TensorTrainGaussian2D, self).__init__()
        self.K = K
        self.M = 2

        # Initialize weights that sum to 1
        Wk0 = np.ones((self.K)) / self.K

        Wk1k0 = np.ones((self.K, self.K))
        Wk1k0 = Wk1k0 / np.sum(Wk1k0, axis=0)

        Wk2k1 = np.ones((self.K, self.K))
        Wk2k1 = Wk2k1 / np.sum(Wk2k1, axis=0)
        
        # Define weights as tf variables
        self.Wk0 = tf.Variable(Wk0, name="Wk0", dtype=tf.dtypes.float32)
        self.Wk1k0 = tf.Variable(Wk1k0, name="Wk1k0", dtype=tf.dtypes.float32)
        self.Wk2k1 = tf.Variable(Wk2k1, name="Wk2k1", dtype=tf.dtypes.float32)

        self.mu = [
            [
                tf.Variable(
                    tf.random.uniform([],-4, 4), 
                    name="mu_{},{}".format(i, j), 
                    dtype=tf.dtypes.float32
                ) for j in range(self.K)
            ] for i in range(self.K)
        ]

        self.sigma = [
            [
                tf.Variable(
                    0.5,
                    name="sigma_{},{}".format(i, j), 
                    dtype=tf.dtypes.float32
                ) for j in range(self.K)
            ] for i in range(self.K)
        ]

        self.distributions = [
            [tfd.Normal(self.mu[i][j], self.sigma[i][j]) for j in range(self.K)] 
            for i in range(self.K)
        ]
        return None

    def call(self, X):
        likelihoods = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
        
        for k0 in range(self.K):
            mid = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
            for k1 in range(self.K):  
                temp_mid = self.Wk1k0[k1, k0] * self.distributions[k1][k0].prob(X[:, 0])
                inner = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
                for k2 in range(self.K):
                    inner += self.Wk2k1[k2, k1] * self.distributions[k2][k1].prob(X[:, 1])
                
                mid += temp_mid*inner
            likelihoods += self.Wk0[k0] * mid
        
        log_likelihoods = tfm.log(likelihoods)
        return log_likelihoods

    def normalize_weights(self):
        """ Ensure that weights always sum to 1 """
        self.Wk0 = tf.Variable(
            self.Wk0 / tf.reduce_sum(self.Wk0), 
            name="Wk0", 
            dtype=tf.dtypes.float32
        )
        self.Wk1k0 = tf.Variable(
            self.Wk1k0 / tf.reduce_sum(self.Wk1k0, axis=0), 
            name="Wk1k0", 
            dtype=tf.dtypes.float32)
        self.Wk2k1 = tf.Variable(
            self.Wk2k1 / tf.reduce_sum(self.Wk2k1, axis=0), 
            name="Wk2k1", 
            dtype=tf.dtypes.float32
        )
        return None

    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
    
        # Compute gradients
        tvars = self.trainable_variables
        gradients = tape.gradient(loss_value, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        self.normalize_weights()
        return loss_value
