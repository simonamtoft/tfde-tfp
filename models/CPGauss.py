import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
import time
from tqdm import tqdm
import numpy as np
tfd = tfp.distributions   
tfm = tf.math

class CPGaussian(tf.keras.Model):
    def __init__(self, K, M, seed = None):
        super(CPGaussian, self).__init__()
        """A CP Decomposition (Diagonal Gaussian Mixture Model) consisting of independent univariate Gaussians

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
        
        return None
    
    def call(self, data):
        if data.shape[1] != self.M:
            raise Exception('Data has wrong dimensions')
            
            
        # Go from logits -> weights
        W = tf.nn.softmax(self.W_logits, axis=1) # axis 1 as it is (1, K)
        
        # Go from raw values -> strictly positive values (ReLU approx.)
        sigma = tfm.softplus(self.sigma)
        
        product = W
        for i in range(self.M):
          result = tfm.exp(tfd.Normal(self.mu[:,i], sigma[:,i]).log_prob(
                  data[:, tf.newaxis, i]
              ))
          
          product = tf.multiply(product,result)
          
        likelihoods = tf.squeeze(tf.reduce_sum(product, axis=1))
        
        # add small number to avoid nan
        log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
        return log_likelihoods
    
    def init_mu(self, data, mode = 'kmeans'):
        """ Initializes the means
        mode = 'kmeans' : Initialize using KMmeans algorithm
        mode = 'random' : Initialize using random
        """
        if mode == 'kmeans':
            kmeans = KMeans(n_clusters=self.K).fit(data)
            mu_kmeans = kmeans.cluster_centers_
            self.mu = tf.Variable(mu_kmeans, name="mu", dtype=tf.dtypes.float32)
        elif mode == 'random':
            mins = np.min(data,axis=0)
            maxs = np.max(data,axis=0)  

            mu_rand = np.random.uniform(mins,maxs, size=(self.K, self.M))
            self.mu = tf.Variable(mu_rand, name="mu", dtype=tf.dtypes.float32)
        else:
            raise Exception('Specified mu initialization not valid')
        return None
    
    @tf.function
    def train_step(self, data, optimizer):
        tvars = self.trainable_variables
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss_value = -tf.reduce_mean(log_likelihoods)
            # Compute gradients
            gradients = tape.gradient(loss_value, tvars)
            
        optimizer.apply_gradients(zip(gradients, tvars))
        return loss_value
    
    def fit(self, dataset, EPOCHS=200, optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),mu_init='kmeans'):
        """ Fits model to a dataset """
        # Initialize mu
        # This is really ugly right now and could be fixed
        for x in (dataset): 
            break
        self.init_mu(x, mode = mu_init)
    
        losses = []
        start_time = time.time()
        for epoch in tqdm(range(EPOCHS),desc='Training CP'):    
            loss = 0
            for i,x in enumerate(dataset):
                loss += self.train_step(x,optimizer) 
            losses.append(loss.numpy()/len(dataset))
                
        end_time = time.time()
        print(f'Training time elapsed: {int(end_time-start_time)} seconds')
        print(f'Final loss: {losses[-1]}')
    
        return losses
    def sample(self, N):
        # TO-DO
        # Simply sample from categorical distributions based on wk0 and W logits
        # And then sample from corresponding distributions.
        pass
    
    def n_parameters(self):
        """ Returns the number of parameters in model """
        n_params = 0
        n_params += self.K # From W_logits
        n_params += 2*self.K*self.M # From mu and sigma

        return n_params
    

#%% Old version

# class CPGaussian(tf.keras.Model):
#     """A CP Decomposition (Diagonal Gaussian Mixture Model) consisting of independent univariate Gaussians

#     Parameters
#     ----------
#     K : int > 0
#     Amount of components
#     M : int > 0
#     Amount of dimensions pr. component
#     """
#     def __init__(self, K, M, data = None, seed = None):
#         super(CPGaussian, self).__init__()
#         self.K = K
#         self.M = M
        
#         if seed != None:
#             tf.random.set_seed(seed)
        
#         self.w_logits = tf.Variable([1.]*K, name="logits")
        
#         if np.all(data != None):
#             # Use Kmeans to get initial guess of mu
#             kmeans = KMeans(n_clusters=K).fit(data)
#             mu_kmeans = kmeans.cluster_centers_
#             self.locs = [
#             [
#                 tf.Variable(
#                     mu_kmeans[i,j], 
#                     name="mu_{},{}".format(i,j)
#                 ) for j in range(self.M)
#             ] for i in range(self.K)]
#         else:
#             self.locs = [
#                 [
#                     tf.Variable(
#                         tf.random.uniform([],-4, 4), 
#                         name="mu_{},{}".format(i,j)
#                     ) for j in range(self.M)
#                 ] for i in range(self.K)
#             ]
            
#         self.scales = [
#             [
#                 tf.Variable(
#                     0.5,
#                     name="sigma_{},{}".format(i,j)
#                 ) for j in range(self.M)
#             ] for i in range(self.K)
#         ]
#         self.distributions = [
#             [
#                 tfd.Normal(self.locs[i][j], self.scales[i][j]) for j in range(M)
#             ] for i in range(K)
#         ]

#         self.components = [
#             tfd.Blockwise(dists) 
#             for dists in self.distributions
#         ]
#         self.density = tfd.Mixture(
#             cat=tfd.Categorical(logits=self.w_logits),
#             components=self.components
#         )
#         return None

#     def call(self, x):
#         log_likelihoods = self.density.log_prob(x)
#         return log_likelihoods

#     def initialize(self, centroids, scales):
#         for i in range(self.K):
#             for j in range(self.M):
#                 self.locs[i][j].assign(centroids[i][j])
#                 self.scales[i][j].assign(scales[i][j])

#     # @tf.function
#     def train_step(self, data, optimizer):
#         with tf.GradientTape() as tape:
#             log_likelihoods = self(data)
#             loss = -tf.reduce_mean(log_likelihoods)
#         tvars = self.trainable_variables
#         gradients = tape.gradient(loss, tvars)
#         optimizer.apply_gradients(zip(gradients, tvars))
#         return loss
#     def fit(self, dataset, EPOCHS=200, optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)):
#         """ Fits model to a dataset """    
#         losses = []
#         start_time = time.time()
#         for epoch in tqdm(range(EPOCHS),desc='Training CP'):    
#             loss = 0
#             for i,x in enumerate(dataset):
#                 loss += self.train_step(x,optimizer) 
#             losses.append(loss.numpy()/len(dataset))
                
#         end_time = time.time()
#         print(f'Training time elapsed: {int(end_time-start_time)} seconds')
#         print(f'Final loss: {losses[-1]}')
    
#         return losses

