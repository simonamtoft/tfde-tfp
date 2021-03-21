import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tf.math


class TensorTrainGaussian2D(tf.keras.Model):
    def __init__(self, K, seed=None):
        super(TensorTrainGaussian2D, self).__init__()
        self.K = K
        self.M = 2
        
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize weights
        Wk0 = np.ones((self.K))
        Wk1k0 = np.ones((self.K, self.K))
        Wk2k1 = np.ones((self.K, self.K))

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
        
        dist = []
        for i in range(self.K):
            for j in range(self.K):
                dist.append(tfd.Normal(self.mu[i][j], self.sigma[i][j]))
        self.joint = tfd.JointDistributionSequential(dist)

        return None

    def call(self, X):
        # likelihoods = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
        Wk0 = tf.nn.softmax(self.Wk0)
        Wk1k0 = tf.nn.softmax(self.Wk1k0, axis=0)
        Wk2k1 = tf.nn.softmax(self.Wk2k1, axis=0)

        d1 = []
        d2 = []
        # [d1.append(X[:, 0]) for s in range(self.K**2)]
        # [d2.append(X[:, 1]) for s in range(self.K**2)]
        d1 = X[:, 0]
        d2 = X[:, 1]

        A = Wk1k0
        B = np.reshape(self.joint.log_prob_parts(d1), (self.K, self.K, -1))
        C = Wk2k1
        D = np.reshape(self.joint.log_prob_parts(d2), (self.K, self.K, -1))
        
        res1 = tf.multiply(A[:, :], B)
        res2 = tf.multiply(C[:, :], D)
        res = tf.tensordot(res1, tf.transpose(res2), axes=2)
        z = Wk0

        likelihoods = tf.reduce_sum(tf.tensordot(z, res, axes=1))
        return tfm.log(likelihoods + np.finfo(np.float32).eps)
        # add small number to avoid nan
        # log_likelihoods = tfm.log(likelihoods+np.finfo(np.float64).eps)
        # return log_likelihoods

    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
    
        # Compute gradients
        tvars = self.trainable_variables
        gradients = tape.gradient(loss_value, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        return loss_value
    
    def get_params(self):   
        """ Get hyper-parameters in numpy-format """
        params = {}
        params['Wk0'] = tf.nn.softmax(self.Wk0).numpy()
        params['Wk1k0'] = tf.nn.softmax(self.Wk1k0, axis=0).numpy()
        params['Wk2k1'] = tf.nn.softmax(self.Wk2k1, axis=0).numpy()
        
        mu = np.zeros((self.K,self.K))
        sigma = np.zeros((self.K,self.K))
        for i in range(self.K):
            for j in range(self.K):
                mu[i,j] = self.mu[i][j].numpy()
                sigma[i,j] = self.sigma[i][j].numpy()
        params['mu'] = mu
        params['sigma'] = sigma
        return params

class TensorTrainGaussian(tf.keras.Model):
    def __init__(self, K, M, seed=None):
        super(TensorTrainGaussian, self).__init__()
        """ M-dimensional Tensor Train gmm model """
        
        self.K = K
        self.M = M       
        
        # To produce reproducible results
        if seed != None:
            tf.random.set_seed(seed)
            
        
        # Initialize weights
        Wk0 = np.ones((1,self.K))      
        self.wk0_logits = tf.Variable(Wk0, name="Wk0_logits", dtype=tf.dtypes.float32)
        W_logits = np.ones((self.M, self.K, self.K))
        self.W_logits = tf.Variable(W_logits, name="W_logits", dtype=tf.dtypes.float32)
        mu = np.random.uniform(-4,4,(self.M, self.K, self.K))
        self.mu = tf.Variable(mu, name="mu", dtype=tf.dtypes.float32)
        pre_sigma = np.random.uniform(0,5,(self.M, self.K, self.K))
        self.pre_sigma = tf.Variable(pre_sigma, name="sigma", dtype=tf.dtypes.float32)


        # Ks = [self.K]
        # [Ks.append(self.K) for i in range(self.M)]
        # self.Ks = Ks
        
        # self.wk0_logits = tf.Variable([[1.]*self.Ks[0]], name='wk0_logits')     
        
        # self.W_logits = [
        #   tf.Variable(
        #     [ [1.]*self.Ks[i] ]*self.Ks[i+1], # (Ks[i+1], Ks[i])
        #     name='W{}_logits'.format(i+1)) 
        #   for i in range(self.M)
        # ]

        # self.mu = [
        #   tf.Variable(
        #     tf.random.uniform([self.Ks[i+1], self.Ks[i]], -5, 5),
        #     name='mu{}'.format(i+1)
        #   ) for i in range(self.M)
        # ]

        # self.pre_sigma = [
        #   tf.Variable(
        #     tf.random.uniform([self.Ks[i+1], self.Ks[i]], 0, 5),
        #     name='sigma{}'.format(i+1)
        #   ) for i in range(self.M)
        # ]

    def call(self, X):
        """ Calculated the log-likelihood of a (N,M) dataset
        In matrix-multiplication format like z . (A (x) B)^T . (C (x) D)^T ...
        """
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
        log_likelihoods = tfm.log(likelihoods+np.finfo(np.float64).eps)
        
        return log_likelihoods

    def sample(self, N):
        # TO-DO
        # Simply sample from categorical distributions based on wk0 and W logits
        # And then sample from corresponding distributions.
        pass

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
