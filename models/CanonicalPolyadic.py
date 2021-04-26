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
    
    def call(self, data):
        if data.shape[1] != self.M:
            raise Exception('Data has wrong dimensions')
            
        # Go from logits -> weights
        W = tf.nn.softmax(self.W_logits, axis=1) # axis 1 as it is (1, K)
        
        # Go from raw values -> strictly positive values (ReLU approx.)
        sigma = tfm.softplus(self.sigma)
        sigma += np.finfo(np.float32).eps # Add small value for numerical stability
        
        product = tfm.log(W)
        for i in range(self.M):
          result = tfd.Normal(self.mu[:,i], sigma[:,i]).log_prob(
                  data[:, tf.newaxis, i]
              )
          product = product + result
          
        log_likelihoods = tf.squeeze(tf.reduce_logsumexp(product, axis=1))
        
        # add small number to avoid nan
        #log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
        return log_likelihoods

    def init_parameters(self, dataset, mode='kmeans', N_init=100):
        """ Initializes the means
        mode = 'kmeans' : Initialize using KMmeans algorithm
        mode = 'random' : Initialize using random
        
        and initializes the variance
        """
        # This is really ugly right now and could be fixed
        for data in (dataset): 
            break

        if mode == 'kmeans':
            kmeans = KMeans(n_clusters=self.K).fit(data)
            mu_kmeans = kmeans.cluster_centers_
            self.mu = tf.Variable(mu_kmeans, name="mu", dtype=tf.dtypes.float32)
        elif mode == 'random':
            # Find the limits of the means
            means_min = np.min(data,axis=0)
            means_max = np.max(data,axis=0)  
        else:
            raise Exception('Specified mu initialization not valid')

        # Find the limits of the variance
        std_max = np.std(data,axis=0)
        
        # Initialize parameter arrays
        pre_sigmas = np.random.uniform(0, std_max, (N_init, self.K, self.M))
        if mode == 'random':
            means = np.random.uniform(means_min,means_max, size=(N_init, self.K, self.M))
        
        # Initialize score array
        score = np.zeros((N_init))
        
        for i in range(N_init):
            self.sigma = tf.Variable(pre_sigmas[i], name="sigma", dtype=tf.dtypes.float32)
            if mode == 'random':
                self.mu = tf.Variable(means[i], name="mu", dtype=tf.dtypes.float32)
            loss_value = -tf.reduce_mean(self(data)).numpy()
            score[i] = loss_value
        idx = np.argmax(score) # Index of best performing set
        
        # Set initial best values of parameters
        self.sigma = tf.Variable(pre_sigmas[idx], name="sigma", dtype=tf.dtypes.float32)
        if mode == 'random':
                self.mu = tf.Variable(means[idx], name="mu", dtype=tf.dtypes.float32)
        
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
    
    def fit(self, dataset, EPOCHS=200, optimizer=None, mu_init='kmeans',
            mute=False, N_init=100, tolerance=1e-7,earlyStop=True):
        """ Fits model to a dataset """
        # Initialize parameters
        self.init_parameters(dataset, mode = mu_init, N_init = N_init)
        
        if optimizer == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        losses = []
        # start_time = time.time()
        for epoch in tqdm(range(EPOCHS), desc='Training CP', disable=mute):    
            loss = 0
            for _, x in enumerate(dataset):
                loss += self.train_step(x, optimizer) 
            losses.append(loss.numpy()/len(dataset))
            if (epoch > 3) and (abs(losses[-2]-losses[-1]) < tolerance):
                break
                
        # end_time = time.time()
        # if not mute:
        #     print(f'Training time elapsed: {int(end_time-start_time)} seconds')
        #     print(f'Final loss: {losses[-1]}')
        losses = np.array(losses)    
        return losses
    def fit_val(self, dataset_train, dataset_val, epochs=200, optimizer=None, mute=False,
            N_init = 100,tolerance=1e-6,N_CONVERGENCE = 5, mu_init='kmeans'):
        """Fits model to a training dataset and validated on a validation dataset
        Input
            dataset     (tf.dataset)            :   The training data to fit the model on.
                                                    Has to be converted to a TF dataset.
            epochs      (int)                   :   The number of epochs to train over.
            optimizer   (tf.keras.optimizers)   :   The optimizer used for training the model.
                                                    Default is the Adam optimizer with lr=1e-3
            mute        (bool)                  :   Whether to time and print after training or not.
            N_CONVERGENCE(int)                  :   Number of values to look at for convergence
        Return
            losses      (array)                 :   Array of the loss after each epoch
        """
        
        if optimizer == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
          
        # Initialize parameters
        self.init_parameters(dataset_train, mode = mu_init, N_init = N_init)
        
        # Get batch size of validation dataset
        batch_size_val = next(iter(dataset_val)).shape[0]
        validation_size = 0
        for x in dataset_val:
            validation_size += x.shape[0]

        losses_train = []
        losses_val = []
        for epoch in tqdm(range(epochs), desc='Training TT', disable=mute,position=0,leave=True):    
            loss = 0
            for _, x in enumerate(dataset_train):
                loss += self.train_step(x, optimizer) 
            losses_train.append(loss.numpy() / len(dataset_train))
            
            # Iterate over validation set
            loss_val = np.zeros(validation_size,dtype=np.float32)
            for j,x in enumerate(dataset_val):
                loss_val[j*batch_size_val:j*batch_size_val+x.shape[0]] = self(x).numpy()
            losses_val.append(-np.mean(loss_val))
            
            # Check the last 4 iterations
            diff = np.zeros((N_CONVERGENCE,))
            for j,lists in enumerate(zip(losses_val[-(N_CONVERGENCE+1):-1], losses_val[-N_CONVERGENCE:])):
                diff[j] = lists[0]-lists[1]
            
            condition1 = all(abs(diff) < tolerance)
            condition2 = all(diff < 0)  
            if (epoch > N_CONVERGENCE+10) and (condition1 or condition2):
                break

        losses_train = np.array(losses_train)
        losses_val = np.array(losses_val)
        return losses_train, losses_val

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

        # Check that trainable params = actual number of parameters
        n_params2 = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        if n_params2 != n_params:
            raise Exception("Number of parameters doens't fit with trainable parameters") 
        return n_params
    def convert_to_bits_dim(self, neg_log_likelihood):
        """ Converts negative log-likelihood to bits/dim
        Used for image datasets (MNIST and CIFAR10)
        """
        bits_dim = -((-neg_log_likelihood/(self.M))-np.log(256))/np.log(2)
        return bits_dim


class CPGeneral(tf.keras.Model):
    def __init__(self, K, dists, params, mod, seed = None):
        """ M-dimensional Tensor Train with Gaussian Mixture Models 
        
        Input
            K       (int)   :   Number of components pr. mixture
            dists   (list)  :   List of M distributions
            params  (list)  :   List of M lists of parameters
            mod     (dict)  :   Dict which dimensions need modifiers (e.g. softmax, softplus)
            seed    (int)   :   Set to other than none in order to reproduce results
        """
        super(CPGeneral, self).__init__()
        self.K = K
        self.M = len(dists)
        self.dists  = dists
        self.params = [ [tf.Variable(p, dtype=tf.float32) for p in lp] for lp in params ]
        self.mod = mod
        W_logits = np.ones((1, self.K))      
        self.W_logits = tf.Variable(W_logits, name="W_logits", dtype=tf.dtypes.float32)
            
    def call(self, data):
        if data.shape[1] != self.M:
            raise Exception('Data has wrong dimensions')
            
        # Go from logits -> weights
        W = tf.nn.softmax(self.W_logits, axis=1) # axis 1 as it is (1, K)
        
        params = [
            [
                [
                    self.params[m][l] 
                    if not (m in self.mod and l in self.mod[m])
                    else 
                    self.mod[m][l](self.params[m][l])
                ] for l in range(len(self.params[m])) # for each list of parameters
            ] for m in range(self.M) # for each dimension
        ]
        
        product = tfm.log(W)
        for i in range(self.M):
          result = self.dists[i](*params[i]).log_prob(
                  data[:, tf.newaxis, i]
              )
          product = product + result
          
        log_likelihoods = tf.squeeze(tf.reduce_logsumexp(product, axis=1))
        
        # add small number to avoid nan
        #log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)
        return log_likelihoods

    def init_parameters(self, dataset, mode='kmeans', N_init=100):
        pass
    
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
    
    def fit(self, dataset, EPOCHS=200, optimizer=None, mu_init='kmeans',
            mute=False, N_init=100):
        """ Fits model to a dataset """
        # Initialize parameters
        self.init_parameters(dataset, mode=mu_init, N_init=N_init)
        
        if optimizer == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        losses = []
        start_time = time.time()
        for epoch in tqdm(range(EPOCHS), desc='Training CPGeneral', disable=mute):    
            loss = 0
            for _, x in enumerate(dataset):
                loss += self.train_step(x,optimizer) 
            losses.append(loss.numpy()/len(dataset))
                
        end_time = time.time()
        if not mute:
            print(f'Training time elapsed: {int(end_time-start_time)} seconds')
            print(f'Final loss: {losses[-1]}')
        return losses

    def sample(self, N):
        # TO-DO
        # Figure out if there's some way to do this without loops...

        samples = []
        params = [
            [
                [
                    self.params[m][l] 
                    if not (m in self.mod and l in self.mod[m])
                    else 
                    self.mod[m][l](self.params[m][l])
                ] for l in range(len(self.params[m])) # for each list of parameters
            ] for m in range(self.M) # for each dimension
        ]

        for _ in range(N):
            sample = []
            ks = tfd.Categorical(logits=self.W_logits[0]).sample()
            for m in range(self.M):
                p = [[p[ks] for p in ps] for ps in params[m]]
                sample.append(
                    self.dists[m](*p).sample()
                )

            samples.append(np.squeeze(np.array(sample)))
        return np.array(samples)
    
    def n_parameters(self):
        """ Returns the number of parameters in model """
        # n_params = 0
        # n_params += self.K # From W_logits
        # n_params += 2*self.K*self.M # From mu and sigma

        # Check that trainable params = actual number of parameters
        n_params2 = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        # if n_params2 != n_params:
        #     raise Exception("Number of parameters doens't fit with trainable parameters")
            
        return n_params2
    

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

