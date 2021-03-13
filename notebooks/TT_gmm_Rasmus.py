import sys, os
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
import utils as utl
tfd = tfp.distributions
tfm = tf.math
import data as d

N = 3000
data = d.gen_8gaussians(batch_size=N)
data = data.astype(np.float32)

# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(f'Data with {N} points')
plt.show()


#%% Set parameters
K = 3 # Number of components


#%% Unit test (For K = 3)
#Initialize parameters randomly
np.random.seed(2)

Wk1k0 = np.random.rand(K,K)
Wk1k0 = Wk1k0/np.sum(Wk1k0,axis=0)

Wk2k1 = np.random.rand(K,K)
# Wk2k1 = Wk2k1/np.sum(Wk2k1)
Wk2k1 = Wk2k1/np.sum(Wk2k1,axis=0)

Wk0 = np.random.rand(K)
Wk0 = Wk0/np.sum(Wk0)
sigma = np.random.uniform(0,3,(K,K))

mu = np.random.uniform(-4,4,(K,K))

# Set number of points and limits
n_points = 1000
lim = 10

x,dx = np.linspace(-lim,lim,n_points,retstep=True)
y,dy = np.linspace(-lim,lim,n_points,retstep=True)
x_grid, y_grid = np.meshgrid(x, y)
X = np.array([x_grid.ravel(), y_grid.ravel()]).T

p = np.zeros((X.shape[0]))
for k0 in range(K):
    
    mid = np.zeros((X.shape[0]))
    for k1 in range(K):
        temp_mid = Wk1k0[k1,k0]*tfd.Normal(mu[k1,k0],sigma[k1,k0]).prob(X[:,0]).numpy()
        
        inner = np.zeros((X.shape[0]))
        for k2 in range(K):
            inner += Wk2k1[k2,k1]*tfd.Normal(mu[k2,k1],sigma[k2,k1]).prob(X[:,1]).numpy()
            
        mid += temp_mid*inner
    p += Wk0[k0]*mid
    

# Show density
plt.imshow(
    p.reshape(n_points,n_points),
    extent=(-lim, lim, -lim, lim),
    origin='lower'
)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Likelihood')
plt.plot()
plt.show()

integrand = np.sum(p)*dx*dy
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')


#%%

class TT_GMM(tf.keras.Model):
    
    def __init__(self, K):
        super(TT_GMM, self).__init__()
        self.K = K
        self.M = 2 # Hard-coded for 2 dimensions
        
        
        Wk0 = np.ones((K))/K
        Wk1k0 = np.ones((K,K))
        Wk1k0 = Wk1k0/np.sum(Wk1k0,axis=0)
        
        Wk2k1 = np.ones((K,K))
        Wk2k1 = Wk2k1/np.sum(Wk2k1,axis=0)        
        
        self.Wk0 = tf.Variable(Wk0,name="Wk0",dtype=tf.dtypes.float32)
        self.Wk1k0 = tf.Variable(Wk1k0,name="Wk1k0",dtype=tf.dtypes.float32)
        self.Wk2k1 = tf.Variable(Wk2k1,name="Wk2k1",dtype=tf.dtypes.float32)
        
        self.mu = [
          [
            tf.Variable(
              tf.random.uniform([],-4, 4), 
              name="mu_{},{}".format(i,j),dtype=tf.dtypes.float32
            ) for j in range(self.K)
          ] for i in range(self.K)
        ]

        self.sigma = [
          [
            tf.Variable(
              0.5,
              name="sigma_{},{}".format(i,j),dtype=tf.dtypes.float32
            ) for j in range(self.K)
          ] for i in range(self.K)
        ]
        
        self.distributions = [
          [
            tfd.Normal(self.mu[i][j], self.sigma[i][j]) for j in range(K)
          ] for i in range(K)
        ]
        return None

    def call(self, X):
        
        likelihoods = tf.zeros((X.shape[0]),dtype=tf.dtypes.float32)
        for k0 in range(self.K):
            
            mid = tf.zeros((X.shape[0]),dtype=tf.dtypes.float32)
            for k1 in range(self.K):  
                temp_mid = self.Wk1k0[k1,k0]*self.distributions[k1][k0].prob(X[:,0])
                
                inner = tf.zeros((X.shape[0]),dtype=tf.dtypes.float32)
                for k2 in range(self.K):
                    inner += self.Wk2k1[k2,k1]*self.distributions[k2][k1].prob(X[:,1])
                    
                mid += temp_mid*inner
            likelihoods += self.Wk0[k0]*mid
        
        
        log_likelihoods = tfm.log(likelihoods)
        return log_likelihoods
    
    def normalizeWeights(self):
        """ Normalizes the weights to always sum to 1
        """
        self.Wk0 = self.Wk0/tf.reduce_sum(self.Wk0)
        self.Wk2k1 = self.Wk2k1/tf.reduce_sum(self.Wk2k1,axis=0)
        self.Wk1k0 = self.Wk1k0/tf.reduce_sum(self.Wk1k0,axis=0)
        
        return None
  
model = TT_GMM(K)

#%% Train model (This doesn't work right now)
@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        log_likelihoods = model(data)
        loss = -tf.reduce_mean(log_likelihoods)
    tvars = model.trainable_variables
    gradients = tape.gradient(loss, tvars)
    optimizer.apply_gradients(zip(gradients, tvars))

# Fit the model
EPOCHS = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
losses = []
for epoch in range(EPOCHS):
    train_step(data)
    
    model.normalizeWeights()
    log_likelihoods = model(data)
    loss = (-np.mean(log_likelihoods))
    losses.append(loss)
    if epoch % 100 == 0:
        print("{}/{} mean neg log likelihood: {}".format(epoch, EPOCHS, loss))
   
        
#%% Plot result
# Set number of points and limits
n_points = 1000
lim = 4

x,dx = np.linspace(-lim,lim,n_points,retstep=True)
y,dy = np.linspace(-lim,lim,n_points,retstep=True)
x_grid, y_grid = np.meshgrid(x, y)
X = np.array([x_grid.ravel(), y_grid.ravel()]).T



p_log = model(X).numpy()
p = np.exp(p_log)

        
# Show density
plt.imshow(
    p.reshape(n_points,n_points),
    extent=(-lim, lim, -lim, lim),
    origin='lower'
)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Likelihood')
plt.plot()
plt.show()


integrand = np.sum(p)*dx*dy
print(f'Density integrates to {round(integrand,4)}')
print('It should be = 1.0')