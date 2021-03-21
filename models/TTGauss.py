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

#%%
class TensorTrainGaussian(tf.keras.Model):
    def __init__(self, K, M, seed=None):
        super(TensorTrainGaussian, self).__init__()
        self.K = K
        self.M = M
        
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize weights
        Wk0 = np.ones((self.K))      
        W = np.ones((self.M, self.K, self.K))

        
        # Define weights as tf variables
        self.Wk0 = tf.Variable(Wk0, name="Wk0", dtype=tf.dtypes.float32)
        self.W = tf.Variable(W, name="W", dtype=tf.dtypes.float32)
        
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
                dist.append(tfd.Normal(self.mu[i][j],self.sigma[i][j]))
        self.joint = tfd.JointDistributionSequential(dist)
        return None

    def call(self, X):
        
        # Softmax out the weights
        Wk0 = tf.nn.softmax(self.Wk0)
        W = tf.nn.softmax(self.W, axis=1)
        
        # Set z
        z = Wk0.numpy()

        for i in range(self.M):
            d = []
            [d.append(X[:,i]) for s in range(self.K**2)]
            
            # Get weights and probabilities for dimension
            A = W[i].numpy()
            B = np.reshape(self.joint.prob_parts(d),(self.K, self.K, -1))
            
            # Perform element-wise multiplication
            res = np.multiply(A[:,:, np.newaxis], B)
            
            # Save product
            if i == 0:
                products = res.T
            else:
                products = products @ res.T
        
        
        likelihoods = np.sum(z @ products, axis = 1)
        
        likelihoods = tf.convert_to_tensor(likelihoods,dtype=tf.float32)
        
        # add small number to avoid nan
        log_likelihoods = tfm.log(likelihoods+np.finfo(np.float64).eps)
        
        return log_likelihoods


    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
    
        # Compute gradients
        tvars = self.trainable_variables
        gradients = tape.gradient(loss_value, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        return loss_value
