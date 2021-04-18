#%%
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import utils as utl
import datasets as d
import models as m
tfd = tfp.distributions
tfm = tf.math

data = d.gen_checkerboard_d3split(batch_size=20000)
batched = d.to_tf_dataset(data, batch_size=200)
# %%
K = 3
M = data.shape[1]
dists = [tfd.Normal, tfd.Normal, tfd.Normal]
params = [
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ]
]
modifiers = {
    0: {1: tfm.softplus},
    1: {1: tfm.softplus},
    2: {1: tfm.softplus}
}
TT = m.TensorTrainGeneral(K, dists, params, modifiers)


dists = [tfd.Normal, tfd.Normal, tfd.Categorical]
params = [
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.ones((K, K, 3))
    ],
]
modifiers = {
    0: {1: tfm.softplus},
    1: {1: tfm.softplus}
}
HTT = m.TensorTrainGeneral(K, dists, params, modifiers)

K = K**2
dists = [tfd.Normal, tfd.Normal, tfd.Categorical]
params = [
    [
        np.random.uniform(-4, 4, (K, )),
        np.random.uniform(0, 4, (K, ))
    ],
    [
        np.random.uniform(-4, 4, (K, )),
        np.random.uniform(0, 4, (K, ))
    ],
    [
        np.ones((K, 3))
    ],
]
modifiers = {
    0: {1: tfm.softplus},
    1: {1: tfm.softplus}
}

HCP = m.CPGeneral(K, dists, params, modifiers)

#%%
EPOCHS = 250
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
losses = HTT.fit(batched, EPOCHS, optimizer)
losses2 = TT.fit(batched, EPOCHS, optimizer)
losses3 = HCP.fit(batched, EPOCHS, optimizer)
# %%
f, ax = plt.subplots(3, 1, figsize=(9,6))
ax[0].plot(losses)
ax[0].set_title("Heterogeneous Tensor Train")
ax[1].plot(losses3)
ax[1].set_title("Heterogeneous CP Decomposition")
ax[2].plot(losses2)
ax[2].set_title("Gaussian Tensor Train")
f.tight_layout()
f.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Training Iterations")
plt.ylabel("Negative Log-Likelihood pr. sample")


# %%
idxs = []
idxs.append(data[:, 2] == 0)
idxs.append(data[:, 2] == 1)
idxs.append(data[:, 2] == 2)

# check integrand of density
limit = 6
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

print("Heterogeneous Tensor Train")
integrals = []
f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([x_grid.ravel(), y_grid.ravel(), i * np.ones((n_points**2,))]).T

    # Get density
    p_log = HTT(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
print("Sum of categories is {}".format(np.sum(integrals)))

# %%
idxs = []
idxs.append(data[:, 2] == 0)
idxs.append(data[:, 2] == 1)
idxs.append(data[:, 2] == 2)

# check integrand of density
limit = 6
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

print("Heterogeneous CP Decomposition")
integrals = []
f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([x_grid.ravel(), y_grid.ravel(), i * np.ones((n_points**2,))]).T

    # Get density
    p_log = HCP(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
print("Sum of categories is {}".format(np.sum(integrals)))

# %%
integrals = []
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([x_grid.ravel(), y_grid.ravel(), i * np.ones((n_points**2,))]).T

    # Get density
    p_log = TT(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
# %%
HTT_samples = HTT.sample(1000)
TT_samples = TT.sample(1000)
HCP_samples = HCP.sample(1000)
# %%
plt.plot(HTT_samples[HTT_samples[:, 2] == 0, 0], HTT_samples[HTT_samples[:, 2] == 0, 1], '.')
plt.plot(HTT_samples[HTT_samples[:, 2] == 1, 0], HTT_samples[HTT_samples[:, 2] == 1, 1], '.')
plt.plot(HTT_samples[HTT_samples[:, 2] == 2, 0], HTT_samples[HTT_samples[:, 2] == 2, 1], '.')
plt.show()
# %%
plt.plot(HCP_samples[HCP_samples[:, 2] == 0, 0], HCP_samples[HCP_samples[:, 2] == 0, 1], '.')
plt.plot(HCP_samples[HCP_samples[:, 2] == 1, 0], HCP_samples[HCP_samples[:, 2] == 1, 1], '.')
plt.plot(HCP_samples[HCP_samples[:, 2] == 2, 0], HCP_samples[HCP_samples[:, 2] == 2, 1], '.')
plt.show()
# %%
TT_samples[:, 2] = np.around(TT_samples[:, 2])
plt.plot(TT_samples[TT_samples[:, 2] == 0, 0], TT_samples[TT_samples[:, 2] == 0, 1], '.')
plt.plot(TT_samples[TT_samples[:, 2] == 1, 0], TT_samples[TT_samples[:, 2] == 1, 1], '.')
plt.plot(TT_samples[TT_samples[:, 2] == 2, 0], TT_samples[TT_samples[:, 2] == 2, 1], '.')
plt.show()
# %%
## Try out having category first

data1 = data[:, [2,0,1]]
batched = d.to_tf_dataset(data1, batch_size=200)

K = 3
M = data.shape[1]
dists = [tfd.Normal, tfd.Normal, tfd.Normal]
params = [
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ]
]
modifiers = {
    0: {1: tfm.softplus},
    1: {1: tfm.softplus},
    2: {1: tfm.softplus}
}
TT = m.TensorTrainGeneral(K, dists, params, modifiers)


dists = [tfd.Categorical, tfd.Normal, tfd.Normal]
params = [
    [
        np.ones((K, K, 3))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
    [
        np.random.uniform(-4, 4, (K, K)),
        np.random.uniform(0, 4, (K, K))
    ],
]
modifiers = {
    1: {1: tfm.softplus},
    2: {1: tfm.softplus}
}
HTT = m.TensorTrainGeneral(K, dists, params, modifiers)

K = K**2
dists = [tfd.Categorical, tfd.Normal, tfd.Normal]
params = [
    [
        np.ones((K, 3))
    ],
    [
        np.random.uniform(-4, 4, (K, )),
        np.random.uniform(0, 4, (K, ))
    ],
    [
        np.random.uniform(-4, 4, (K, )),
        np.random.uniform(0, 4, (K, ))
    ],
]
modifiers = {
    1: {1: tfm.softplus},
    2: {1: tfm.softplus}
}

HCP = m.CPGeneral(K, dists, params, modifiers)

#%%
EPOCHS = 250
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
losses = HTT.fit(batched, EPOCHS, optimizer)
losses2 = TT.fit(batched, EPOCHS, optimizer)
losses3 = HCP.fit(batched, EPOCHS, optimizer)
# %%
f, ax = plt.subplots(3, 1, figsize=(9,6))
ax[0].plot(losses)
ax[0].set_title("Heterogeneous Tensor Train")
ax[1].plot(losses3)
ax[1].set_title("Heterogeneous CP Decomposition")
ax[2].plot(losses2)
ax[2].set_title("Gaussian Tensor Train")
f.tight_layout()
f.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Training Iterations")
plt.ylabel("Negative Log-Likelihood pr. sample")


# %%
idxs = []
idxs.append(data[:, 2] == 0)
idxs.append(data[:, 2] == 1)
idxs.append(data[:, 2] == 2)

# check integrand of density
limit = 6
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

print("Heterogeneous Tensor Train")
integrals = []
f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([i * np.ones((n_points**2,)), x_grid.ravel(), y_grid.ravel()]).T

    # Get density
    p_log = HTT(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
print("Sum of categories is {}".format(np.sum(integrals)))

# %%
idxs = []
idxs.append(data[:, 2] == 0)
idxs.append(data[:, 2] == 1)
idxs.append(data[:, 2] == 2)

# check integrand of density
limit = 6
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

print("Heterogeneous CP Decomposition")
integrals = []
f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([ i * np.ones((n_points**2,)), x_grid.ravel(), y_grid.ravel(),]).T

    # Get density
    p_log = HCP(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
print("Sum of categories is {}".format(np.sum(integrals)))

# %%
integrals = []
n_points = 300
x, dx = np.linspace(-limit, limit, n_points, retstep=True)
y, dy = np.linspace(-limit, limit, n_points, retstep=True)

x_grid, y_grid = np.meshgrid(x, y)

f, ax = plt.subplots(1, M, figsize=(18,6))
for i in range(M):
    X = np.array([i * np.ones((n_points**2,)), x_grid.ravel(), y_grid.ravel(), ]).T

    # Get density
    p_log = TT(X).numpy()
    p = np.exp(p_log)

    # Compute integrand
    integral = np.sum(p)*dx*dy
    integrals.append(integral)

    #ax[i].plot(data[idxs[i], 0], data[idxs[i], 1], '.',alpha = 0.1)
    #ax[i].contour(x_grid, y_grid, p.reshape(n_points, n_points))
    #ax[i].axis('equal')

    im = ax[i].imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap='viridis'
    )
    cbar = plt.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('inferno')
f.tight_layout()
# %%
HTT_samples = HTT.sample(1000)
TT_samples = TT.sample(1000)
HCP_samples = HCP.sample(1000)
# %%
plt.plot(HTT_samples[HTT_samples[:, 0] == 0, 1], HTT_samples[HTT_samples[:, 0] == 0, 2], '.')
plt.plot(HTT_samples[HTT_samples[:, 0] == 1, 1], HTT_samples[HTT_samples[:, 0] == 1, 2], '.')
plt.plot(HTT_samples[HTT_samples[:, 0] == 2, 1], HTT_samples[HTT_samples[:, 0] == 2, 2], '.')
plt.show()
# %%
plt.plot(HCP_samples[HCP_samples[:, 0] == 0, 1], HCP_samples[HCP_samples[:, 0] == 0, 2], '.')
plt.plot(HCP_samples[HCP_samples[:, 0] == 1, 1], HCP_samples[HCP_samples[:, 0] == 1, 2], '.')
plt.plot(HCP_samples[HCP_samples[:, 0] == 2, 1], HCP_samples[HCP_samples[:, 0] == 2, 2], '.')
plt.show()
# %%
TT_samples[:, 0] = np.around(TT_samples[:, 0])
plt.plot(TT_samples[TT_samples[:, 0] == 0, 1], TT_samples[TT_samples[:, 0] == 0, 2], '.')
plt.plot(TT_samples[TT_samples[:, 0] == 1, 1], TT_samples[TT_samples[:, 0] == 1, 2], '.')
plt.plot(TT_samples[TT_samples[:, 0] == 2, 1], TT_samples[TT_samples[:, 0] == 2, 2], '.')
plt.show()
# %%