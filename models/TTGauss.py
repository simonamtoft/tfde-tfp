import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tf.math


class TensorTrainGaussian2D(tf.keras.Model):
    def __init__(self, K, M=None,seed=None):
        super(TensorTrainGaussian2D, self).__init__()
        self.K = K
        self.M = 2
        
        if seed != None:
            tf.random.set_seed(seed)

        # Initialize weights that sum to 1
        Wk0 = np.ones((self.K)) #/ self.K

        Wk1k0 = np.ones((self.K, self.K))
        #Wk1k0 = Wk1k0 / np.sum(Wk1k0, axis=0)

        Wk2k1 = np.ones((self.K, self.K))
        #Wk2k1 = Wk2k1 / np.sum(Wk2k1, axis=0)
        
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
        Wk0 = tf.nn.softmax(self.Wk0)
        Wk1k0 = tf.nn.softmax(self.Wk1k0, axis=0)
        Wk2k1 = tf.nn.softmax(self.Wk2k1, axis=0)
        for k0 in range(self.K):
            mid = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
            for k1 in range(self.K):  
                temp_mid = Wk1k0[k1, k0] * self.distributions[k1][k0].prob(X[:, 0])
                inner = tf.zeros((X.shape[0]), dtype=tf.dtypes.float32)
                for k2 in range(self.K):
                    inner += Wk2k1[k2, k1] * self.distributions[k2][k1].prob(X[:, 1])
                
                mid += temp_mid*inner
            likelihoods += Wk0[k0] * mid
        
        # add small number to avoid nan
        log_likelihoods = tfm.log(likelihoods+np.finfo(np.float64).eps)
        return log_likelihoods

    def normalize_weights(self):
        """ Ensure that weights always sum to 1 """        
        self.Wk0 = tf.Variable(self.Wk0/tf.reduce_sum(self.Wk0),name="Wk0")
        self.Wk1k0 = tf.Variable(self.Wk1k0/tf.reduce_sum(self.Wk1k0,axis=0),name="Wk1k0")
        self.Wk2k1 = tf.Variable(self.Wk2k1/tf.reduce_sum(self.Wk2k1,axis=0),name="Wk2k1")
        return None

    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
    
        # Compute gradients
        tvars = self.trainable_variables
        gradients = tape.gradient(loss_value, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        #self.normalize_weights()
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
            B = np.reshape(self.joint.prob_parts(d),(self.K,self.K,-1))
            
            # Perform element-wise multiplication
            res = np.multiply(A[:,:,np.newaxis],B)
            
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




