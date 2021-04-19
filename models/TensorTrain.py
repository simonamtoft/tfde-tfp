import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tqdm import tqdm
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

    # def init_parameters(self, ds, N_init=None):
    #     return NotImplementedError

    def fit(self, dataset, epochs=200, optimizer=None, mute=False,
            N_init = 100,tolerance=1e-7):
        """Fits model to a dataset
        Input
            dataset     (tf.dataset)            :   The training data to fit the model on.
                                                    Has to be converted to a TF dataset.
            epochs      (int)                   :   The number of epochs to train over.
            optimizer   (tf.keras.optimizers)   :   The optimizer used for training the model.
                                                    Default is the Adam optimizer with lr=1e-3
            mute        (bool)                  :   Whether to time and print after training or not.
        Return
            losses      (array)                 :   Array of the loss after each epoch
        """
        
        if optimizer == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
          
        # Initialize parameters
        self.init_parameters(dataset, N_init=N_init)

        losses = []
        start_time = time.time()
        for epoch in tqdm(range(epochs), desc='Training TT', disable=mute,position=0,leave=True):    
            loss = 0
            for _, x in enumerate(dataset):
                loss += self.train_step(x, optimizer) 
            losses.append(loss.numpy() / len(dataset))
            if (epoch > 3) and (abs(losses[-2]-losses[-1]) < tolerance):
                break

        end_time = time.time()
        if not mute:
            print(f'Training time elapsed: {int(end_time-start_time)} seconds')
            print(f'Final loss: {losses[-1]}')
        losses = np.array(losses)
        return losses

    def log_space_product_tf(self, A, B):
        """ Multiply matrices in log domain (more stable)
        https://stackoverflow.com/questions/36467022/handling-matrix-multiplication-in-log-space-in-python
        
        input : 
            A = Tensor([N,K,M])
            B = Tensor([N,M,K])
        output :
            C = Tensor([N,K,K])
        """
        Astack = tf.transpose(tf.stack([A]*A.shape[1],axis=1),perm=[0,3,2,1])
        Bstack = tf.transpose(tf.stack([B]*B.shape[2],axis=1),perm=[0,2,1,3])
        C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
        return C

    def log_space_product_vector_tf(self, a, B):
        """ Multiply vector with matrix in log domain (more stable)
        https://stackoverflow.com/questions/36467022/handling-matrix-multiplication-in-log-space-in-python
        
        input : 
            a = Tensor([N,1,K])
            B = Tensor([N,K,K])
        output :
            C = Tensor([N,K])
        """
        Astack = tf.transpose(a,perm=[0,2,1])
        Bstack = tf.transpose(B,perm=[0,1,2])
        C = tfm.reduce_logsumexp(Astack+Bstack, axis=1)
        return C


class TensorTrainGaussian(TensorTrainModel):
    def __init__(self, K, M, seed=None):
        """ M-dimensional Tensor Train with Gaussian Mixture Models 
        
        Input
            K       (int)   :   The number of mixtures the model consits of
            M       (int)   :   The number of dimensions of the data
            seed    (int)   :   Set to other than none in order to reproduce results
        """
        super(TensorTrainGaussian, self).__init__(K, M, seed)
        mu = np.random.uniform(-4, 4, (self.M, self.K, self.K))
        pre_sigma = np.random.uniform(0, 5, (self.M, self.K, self.K))
        # mu = np.random.uniform(-4, 4, (self.K, self.K))
        # pre_sigma = np.random.uniform(0, 5, (self.K, self.K))
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
        wk0 = tf.nn.softmax(self.wk0_logits, axis=1) # axis 1 as it is (1, K0)
        W = [tf.nn.softmax(self.W_logits[i], axis=0) for i in range(self.M)]
        
        # Go from raw values -> strictly positive values (ReLU approx.)
        sigma = [tfm.softplus(self.pre_sigma[i]) for i in range(self.M)]
  
        if self.M < 7:
        # ######### Multiply in exp_domain
            product = tf.eye(wk0.shape[1]) # start out with identity matrix
            for i in range(self.M):
              result = tfm.exp(
                  tfm.log(W[i]) + tfd.Normal(self.mu[i], sigma[i]).log_prob(
                      # Make data broadcastable into (n, km, kn)
                      X[:, tf.newaxis, tf.newaxis, i]
                  )
                #   tfm.log(W[i]) + tfd.Normal(self.mu, sigma).log_prob(
                #       # Make data broadcastable into (n, km, kn)
                #       X[:, tf.newaxis, tf.newaxis, i]
                #   )
              ) # intermediary calculation in log-domain -> exp after.
              # Keep batch dimension in place, transpose matrices
              product = product @ tf.transpose(result, perm=[0, 2, 1])
            
            # In order: Squeeze (n, 1, k_last) -> (n, k_last).
            # Reduce sum over k_last into (n, )
            # Squeeze result to (n, ) if n > 1 or () if n == 1
            likelihoods = tf.squeeze(tf.reduce_sum(tf.squeeze(wk0 @ product, axis=1), axis=1))
            # add small number to avoid nan
            log_likelihoods = tfm.log(likelihoods + np.finfo(np.float64).eps)    
        else:
            ######### Multiply in log_domain  
            # Inner product
            product = tfm.log(W[0]) + tfd.Normal(self.mu[0], sigma[0]).log_prob(
              X[:, tf.newaxis, tf.newaxis, 0])
            product = tf.transpose(product, perm=[0, 2, 1])
            for i in range(1,self.M):
                result = tfm.log(W[i]) + tfd.Normal(self.mu[i], sigma[i]).log_prob(
                      X[:, tf.newaxis, tf.newaxis, i])    
                product = self.log_space_product_tf(product, tf.transpose(result, perm=[0, 2, 1]))
            
            # Multiply with wk0
            prod = self.log_space_product_vector_tf(tf.expand_dims(tfm.log(wk0),axis=1),product)
            log_likelihoods = tf.reduce_logsumexp(prod,axis=1)
        
        return log_likelihoods

    def sample(self, N):
        # TO-DO
        # Simply sample from categorical distributions based on wk0 and W logits
        # And then sample from corresponding distributions.
        pass
    
    def init_parameters(self, dataset,N_init = 100):
        """ Initializes the parameters in the gaussian models 
        to the lowest value of N_init random initializations
        """
        
        for i, x in enumerate(dataset):
            break

        # Find the limits of the parameters
        means_min = np.min(x)
        means_max = np.max(x)
        std_max = np.max(np.std(x,axis=0))
        
        # Initialize parameter arrays
        means = np.random.uniform(means_min, means_max, (N_init, self.M, self.K, self.K))
        pre_sigmas = np.random.uniform(0, std_max, (N_init, self.M, self.K, self.K))
        
        # Initialize score array
        score = np.zeros((N_init))
        
        for i in range(N_init):
            self.mu = tf.Variable(means[i], name="mu", dtype=tf.dtypes.float32)
            self.pre_sigma = tf.Variable(pre_sigmas[i], name="sigma", dtype=tf.dtypes.float32)
            
            loss_value = -tf.reduce_mean(self(x)).numpy()
            score[i] = loss_value
        idx = np.argmax(score) # Index of best performing set


        # Set initial best values of parameters
        self.mu = tf.Variable(means[idx], name="mu", dtype=tf.dtypes.float32)
        self.pre_sigma = tf.Variable(pre_sigmas[idx], name="sigma", dtype=tf.dtypes.float32)
        return None
    
    def n_parameters(self):
        """ Returns the number of parameters in model """
        n_params = 0
        n_params += self.K # From W_k0
        n_params += self.M*self.K*self.K # From W_logits
        n_params += 2*self.M*self.K*self.K # From mu and sigma
        
        # Check that trainable params = actual number of parameters
        n_params2 = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        if n_params2 != n_params:
            raise Exception("Number of parameters doens't fit with trainable parameters")
        
        return n_params


class TensorTrainGeneral(TensorTrainModel):
    def __init__(self, K, dists, params, mod, seed=None):
        """ M-dimensional Tensor Train with Gaussian Mixture Models 
        
        Input
            K       (int)   :   Number of components pr. mixture
            dists   (list)  :   List of M distributions
            params  (list)  :   List of M lists of parameters
            mod     (dict)  :   Dict which dimensions need modifiers (e.g. softmax, softplus)
            seed    (int)   :   Set to other than none in order to reproduce results
        """
        super(TensorTrainGeneral, self).__init__(K, len(dists), seed)
        self.dists  = dists
        self.params = [ [tf.Variable(p, dtype=tf.float32) for p in lp] for lp in params ]
        self.mod = mod

    def fix_params(self):
        return [
            [
                [
                    self.params[m][l] 
                    if not (m in self.mod and l in self.mod[m])
                    else 
                    self.mod[m][l](self.params[m][l])
                ] for l in range(len(self.params[m])) # for each list of parameters
            ] for m in range(self.M) # for each dimension
        ]

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

        # Modify params
        params = self.fix_params()

        if self.M < 7:
        # ######### Multiply in exp_domain
            product = tf.eye(wk0.shape[1]) # start out with identity matrix
            for i in range(self.M):
              result = tfm.exp(
                  tfm.log(W[i]) + self.dists[i](*params[i]).log_prob(
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
        else:
            ######### Multiply in log_domain  
            # Inner product
            product = tfm.log(W[0]) + self.dists[0](*params[0]).log_prob(
              X[:, tf.newaxis, tf.newaxis, 0])
            product = tf.transpose(product, perm=[0, 2, 1])
            for i in range(1,self.M):
                result = tfm.log(W[i]) + self.dists[i](*params[i]).log_prob(
                      X[:, tf.newaxis, tf.newaxis, i])    
                product = self.log_space_product_tf(product, tf.transpose(result, perm=[0, 2, 1]))
            
            # Multiply with wk0
            prod = self.log_space_product_vector_tf(tf.expand_dims(tfm.log(wk0),axis=1),product)
            log_likelihoods = tf.reduce_logsumexp(prod,axis=1)

        return log_likelihoods

    def sample(self, N):
        # TO-DO
        # Figure out if there's some way to do this without loops...

        samples = []
        params = self.fix_params()

        for _ in range(N):
            sample = []
            ks = [tfd.Categorical(logits=self.wk0_logits[0]).sample()]
            for m in range(self.M):
                ks.append(tfd.Categorical(logits=self.W_logits[m,:,ks[m]]).sample())
                p = [[p[ks[m+1], ks[m]] for p in ps] for ps in params[m]]
                sample.append(
                    self.dists[m](*p).sample()
                )

            samples.append(np.squeeze(np.array(sample)))
        return np.array(samples)
    
    def init_parameters(self, dataset,N_init = 100):
        pass
    
    def n_parameters(self):
        """ Returns the number of parameters in model """
        # n_params = 0
        # n_params += self.K # From W_k0
        # n_params += self.M*self.K*self.K # From W_logits
        # n_params += 2*self.K*self.K # From mu and sigma
        
        # Check that trainable params = actual number of parameters
        n_params2 = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        # if n_params2 != n_params:
        #     raise Exception("Number of parameters doens't fit with trainable parameters")
        
        return n_params2

