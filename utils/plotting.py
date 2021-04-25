import numpy as np 
import matplotlib.pyplot as plt


def plot_contours(ax, data, model, limit=4, n_points=1000, alpha=1.0):
    """visualize the different distributions over the data as a contour plot

    Inputs
        ax              :   Axis on which to plot
        model           :   The trained tf.keras.model
        limit (float)   :   The abs limit of the plot in the two dimensions.

    Outputs
        A plot of the contours of the distributions and the dataset points.
    """

    x, _ = np.linspace(-limit, limit, n_points,retstep=True)
    y, _ = np.linspace(-limit, limit, n_points,retstep=True)
    x_grid, y_grid = np.meshgrid(x, y) 
    X = np.array([x_grid.ravel(), y_grid.ravel()]).T

    # Get the likelihood
    p_log = model(X).numpy()
    p = np.exp(p_log)

    # _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(data[:, 0], data[:, 1], '.',alpha = alpha)
    ax.contour(x_grid, y_grid, p.reshape(n_points, n_points))
    ax.axis('equal')
    return None


def plot_density(ax, model, limit=4, n_points=1000, cmap='gray', cbar=True, axis=True, gmm=False):
    """visualize the distribution as a density plot
    
    Inputs
        model           :   The trained tf.keras.model
        limit (float)   :   The abs limit of the plot in the two dimensions.

    Outputs
        A plt.figure that shows the density plot of the cluster distributions.
    """

    # create probability map
    x, dx = np.linspace(-limit, limit, n_points, retstep=True)
    y, dy = np.linspace(-limit, limit, n_points, retstep=True)
    x_grid, y_grid = np.meshgrid(x, y)
    X = np.array([x_grid.ravel(), y_grid.ravel()]).T
    
    # probabilities
    if gmm == True:
        p_log = model.score_samples(X)
    else:
        p_log = model(X).numpy()

    p = np.exp(p_log)

    # sum probs
    integrand = np.sum(p)*dx*dy
            
    # Show density
    im = ax.imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower',
        cmap=cmap
    )

    if not axis:
        ax.axis('off')

    if cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Likelihood')

    return integrand
