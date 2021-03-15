import numpy as np 
import matplotlib.pyplot as plt


def plot_contours(data, model, limit, n_points, title=None):
    """visualize the different distributions over the data as a contour plot

    Inputs
        model           :   The trained tf.keras.model
        limit (float)   :   The abs limit of the plot in the two dimensions.

    Outputs
        A plot of the contours of the distributions and the dataset points.
    """

    x, _ = np.linspace(-limit, limit, n_points,retstep=True)
    y, _ = np.linspace(-limit, limit, n_points,retstep=True)
    x_grid, y_grid = np.meshgrid(x, y) 
    X = np.array([x_grid.ravel(), y_grid.ravel()]).T

    p_log = model(X).numpy()
    p = np.exp(p_log)

    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(data[:, 0], data[:, 1], '.')
    ax.contour(x_grid, y_grid, p.reshape(1000, 1000))
    ax.axis('equal')
    if title != None:
        ax.set_title(title)
    plt.show()
    return None


# THIS MADE FOR TT, NOT CP NOTEBOOK VERSION
def plot_density(model, limit, n_points):
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
    p_log = model(X).numpy()
    p = np.exp(p_log)

    # sum probs
    integrand = np.sum(p)*dx*dy
            
    # Show density
    plt.imshow(
        p.reshape(n_points, n_points),
        extent=(-limit, limit, -limit, limit),
        origin='lower'
    )
    plt.title(f'int(p) = {round(integrand, 4)}')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Likelihood')
    plt.plot()
    plt.show()
    return None