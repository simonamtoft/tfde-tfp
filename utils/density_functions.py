import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions


def unitTest(model, limits=[-1,1], n_points=1000):
    """Integrate a density over a given limit in 2d-space
    To make a unit-test and verify that it is a density
    Should return value close to 1.0

    Inputs
        distribution        : A tfd distribution in 2d
        limits (list)       : Either a list with limits [-X,X] or scalar: X
        n_points (int)      : Number of point to integrate. Increate for precision

    Outputs
        integrand (float)   : Integration over area (should be 1)
    """

    # Construct meshgrid
    if isinstance(limits, list):
        x, dx = np.linspace(limits[0], limits[1], n_points, retstep=True)
        y, dy = np.linspace(limits[0], limits[1], n_points, retstep=True)
    else:
        x, dx = np.linspace(-limits, limits, n_points, retstep=True)
        y, dy = np.linspace(-limits, limits, n_points, retstep=True)
    x_grid, y_grid = np.meshgrid(x, y)
    X = np.array([x_grid.ravel(), y_grid.ravel()]).T
    
    # Get density
    p_log = model(X).numpy()
    p = np.exp(p_log)
    
    # Compute integrand
    integrand = np.sum(p)*dx*dy
    return integrand



def plot_density_3d_paper(model, ax, limit=6, n_points=2000, cmap='gray'):
    # construct meshgrid
    x, dx = np.linspace(-limit, limit, n_points, retstep=True)
    y, dy = np.linspace(-limit, limit, n_points, retstep=True)
    x_grid, y_grid = np.meshgrid(x, y)

    # f, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        X = np.array([x_grid.ravel(), y_grid.ravel(), i * np.ones((n_points**2,))]).T
        # Get density
        p_log = model(X).numpy()
        p = np.exp(p_log)
        im = ax[i].imshow(
            p.reshape(n_points, n_points),
            extent=(-limit, limit, -limit, limit),
            origin='lower',
            cmap=cmap
        )
        ax[i].axis('off')


def plot_density_3d(model, M, title="", limit=6, n_points=300):
    # construct meshgrid
    x, dx = np.linspace(-limit, limit, n_points, retstep=True)
    y, dy = np.linspace(-limit, limit, n_points, retstep=True)
    x_grid, y_grid = np.meshgrid(x, y)

    print(title)
    integrals = []
    f, ax = plt.subplots(1, M, figsize=(15, 5))
    for i in range(M):
        X = np.array([x_grid.ravel(), y_grid.ravel(), i * np.ones((n_points**2,))]).T

        # Get density
        p_log = model(X).numpy()
        p = np.exp(p_log)

        # Compute integrand
        integral = np.sum(p)*dx*dy
        integrals.append(integral)

        im = ax[i].imshow(
            p.reshape(n_points, n_points),
            extent=(-limit, limit, -limit, limit),
            origin='lower',
            cmap='gray'
        )
        cbar = plt.colorbar(im, ax=ax[i])
        cbar.ax.set_ylabel('inferno')
    f.tight_layout()
    print("Sum of categories is {}".format(np.sum(integrals)))



#%% Function examples:
if __name__ == '__main__':
    
    # Initialize a distribution
    dist = tfd.MultivariateNormalDiag(loc=[0,0],scale_diag=[1,1])

    # Compute unit test over a good inverval
    integrand = unitTest(dist,limits=4)
    
    print(f'Integrand is {np.round(integrand,4)}')
