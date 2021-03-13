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

def create_cluster_distributions(means, covs, nc, nf, gpd):
  clusters = []
  for i in range(nc):
    dim_mixtures = []
    for j in range(nf):
      components = []
      for k in range(gpd):
        components.append(
          tfd.Normal(
            means[k][i][j], 
            np.sqrt(covs[k][i][j])
          )
        )
      # Add components to the jth dimension mixture
      dim_mixtures.append(
        tfd.Mixture(
          cat=tfd.Categorical(probs=[1/gpd]*gpd),
          components=components,
        )
      )
    clusters.append(tfd.Blockwise(dim_mixtures))

  return clusters

#%%
# Number of datapoints to generate
N = 3000
data = d.gen_checkerboard(batch_size=N)
data = data.astype(np.float32)


# Inspect the data
f,ax = plt.subplots(figsize=(5,5))
ax.plot(data[:, 0], data[:, 1], '.')
ax.axis('equal')
ax.set_title(f'Data with {N} points')
plt.show()


# Set parameters
N_CLUSTERS = 8              # Number of cluster
GPD = 2                     # gauss per dimension in each cluster distribtuion 
N_FEATURES = data.shape[1]  # the dimensions

#%% Initialize arrays
# Use Kmeans to get initial guess of mu
kmeans = KMeans(n_clusters=N_CLUSTERS).fit(data)
mu_nd = kmeans.cluster_centers_

# use same means for each gauss in same cluster
mu = []
for i in range(GPD):
  mu.append(mu_nd)
mu = np.array(mu)

# Create cov matrix 
covs_nd = np.full(
    (N_CLUSTERS, N_FEATURES),
    np.diag(
        np.cov(data, rowvar=False)
    )
).astype(np.float32)

# same as for means
covs = []
for i in range(GPD):
  covs.append(covs_nd)
covs = np.array(covs)

#%% Training
# Number of epochs to run the loop
epochs = 25

# training loop
pi = 1/N_CLUSTERS * np.ones((N_CLUSTERS))
for epoch in range(epochs):

    clusters = create_cluster_distributions(mu, covs, N_CLUSTERS, N_FEATURES, GPD)

    # total responsibility assigned to each cluster 
    r = np.zeros((N, N_CLUSTERS))
    for i in range(N_CLUSTERS):
        r[:,i] = pi[i]*clusters[i].prob(data).numpy()
    r = r / r.sum(axis=1,keepdims=1)  

    # total responsibility assigned to each cluster
    cluster_weights = r.sum(axis = 0)

    # Update prior
    pi = cluster_weights / N

    # update means
    weighted_sum = np.dot(r.T, data)
    for k in range(GPD):
      mu[k] = (weighted_sum / cluster_weights.reshape(-1, 1)).astype(np.float32)

    # Compute covariances
    for i in range(N_CLUSTERS):
      for k in range(GPD):
        diff = (data - mu[k][i]).T
        weighted_sum = np.dot(r[:, i] * diff, diff.T)
        covs[k][i] = np.diag(weighted_sum / cluster_weights[i])
        
#%% Plot results

utl.plot_contours(data, clusters, 4,'Final distribution', N_CLUSTERS)
utl.plot_density(clusters, 4,'Density', N_CLUSTERS)


# Check if clusters are correct densities
integrands = np.zeros((N_CLUSTERS))
for i in range(N_CLUSTERS): 
    integrands[i] = utl.unitTest(clusters[0],limits=4)

if all(1-integrands < 1e-2):
    print('Densities are correct (their integral is 1)')