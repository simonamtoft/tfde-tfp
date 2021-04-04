import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tqdm import tqdm
tfd = tfp.distributions
tfm = tf.math


class TensorRingModel(tf.keras.Model):
    def __init__(self, K, M, seed=None):
        super(TensorRingModel, self).__init__()
        self.K = K
        self.M = M

        # Enable reproduction of results
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize model weights  
        W_logits = np.ones((self.M, self.K, self.K))

        # Define as TensorFlow variables
        self.W_logits = tf.Variable(W_logits, name="W_logits", dtype=tf.dtypes.float32)
        return None

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
    def fit(self, dataset, EPOCHS=200, optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            mute = False):
        """ Fits model to a dataset """
    
        losses = []
        start_time = time.time()
        for epoch in tqdm(range(EPOCHS),desc='Training TR',disable=mute):    
            loss = 0
            for i,x in enumerate(dataset):
                loss += self.train_step(x,optimizer) 
            losses.append(loss.numpy()/len(dataset))
                
        end_time = time.time()
        if not mute:
            print(f'Training time elapsed: {int(end_time-start_time)} seconds')
            print(f'Final loss: {losses[-1]}')
    
        return losses
        
        


class TensorRingGaussian(TensorRingModel):
    def __init__(self, K, M, seed=None):
        """ M-dimensional Tensor Train with Gaussian Mixture Models 
        
        Inputs
            K       (int)   :   The number of mixtures the model consits of
            M       (int)   :   The number of dimensions of the data
            seed    (int)   :   Set to other than none in order to reproduce results
        """
        super(TensorRingGaussian, self).__init__(K, M, seed)
        # mu = np.random.uniform(-4, 4, (self.M, self.K, self.K))
        # pre_sigma = np.random.uniform(0, 5, (self.M, self.K, self.K))
        mu = np.random.uniform(-4, 4, (self.K, self.K))
        pre_sigma = np.random.uniform(0, 5, (self.K, self.K))
        self.mu = tf.Variable(mu, name="mu", dtype=tf.dtypes.float32)
        self.pre_sigma = tf.Variable(pre_sigma, name="sigma", dtype=tf.dtypes.float32)
        return None

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
        W = [tf.nn.softmax(self.W_logits[i], axis=0) for i in range(self.M)]
        # W = tf.nn.softmax(self.W_logits,axis=2)
        # W[0] = tf.reshape(tf.nn.softmax(tf.reshape(self.W_logits[0],(-1))),(self.K,self.K))
        
        # Go from raw values -> strictly positive values (ReLU approx.)
        sigma = tfm.softplus(self.pre_sigma)
  
        product = tf.eye(self.K) # start out with identity matrix
        for i in range(1,self.M):
          result = tfm.exp(
              tfm.log(W[i]) + tfd.Normal(self.mu, sigma).log_prob(
                  X[:, tf.newaxis, tf.newaxis, i])) 
          product = product @ tf.transpose(result, perm=[0, 2, 1])
        
        result = tfm.exp(tfm.log(W[0]) + tfd.Normal(self.mu, sigma).log_prob(
                  X[:, tf.newaxis, tf.newaxis, 0]))
        product = product @ result
        
        likelihoods = tf.squeeze(tf.reduce_sum(tf.reduce_sum(product,axis=1),axis=1))
  
        # add small number to avoid nan
        log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
        return log_likelihoods

    def sample(self, N):
        # TO-DO
        # Simply sample from categorical distributions based on wk0 and W logits
        # And then sample from corresponding distributions.
        pass
    
    def n_parameters(self):
        """ Returns the number of parameters in model """
        n_params = 0
        n_params += self.M*self.K*self.K # From W_logits
        n_params += 2*self.K*self.K # From mu and sigma

        return n_params

