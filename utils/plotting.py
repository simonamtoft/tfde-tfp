import numpy as np 
import matplotlib.pyplot as plt


def plot_contours(data, distributions, limit, title, nc):
    """visualize the different distributions over the data as a contour plot

    Inputs
        data (np.array) :   The points (x, y) of the dataset to be plotted. 
                            Dimension [nc, 2]
        distributions   :   GGGG
        limit (float)   :   The abs limit of the plot in both x- and y-direction.
        title (string)  :   Title of the plot.
        nc (int)        :   The number of clusters.

    Outputs
        A plot of the contours of the distributions and the dataset points.
    """
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko',alpha=0.1)

    delta = 0.025
    x = np.arange(-limit, limit, delta)
    y = np.arange(-limit, limit, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    for i in range(nc):
        z_grid = distributions[i].prob(coordinates).numpy().reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid)

    plt.title(title)
    plt.tight_layout()
    plt.show()
    return None


def plot_density(distributions, limit, title, nc):
    """visualize the different distributions over the data as a density plot
    
    Inputs
        data (np.array) :   The points (x, y) of the dataset to be plotted. 
                            Dimension [nc, 2]
        distributions   :   GGGG
        limit (float)   :   The abs limit of the plot in both x- and y-direction.
        title (string)  :   Title of the plot.
        nc (int)        :   The number of clusters.

    Outputs
        A plt.figure that shows the density plot of the cluster distributions.
    """
    plt.figure()
    delta = 0.025
    x = np.arange(-limit, limit, delta)
    y = np.arange(-limit, limit, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    I = np.zeros((nc, x_grid.shape[0], x_grid.shape[1]))

    for i in range(nc):
        z_grid = distributions[i].prob(coordinates).numpy().reshape(x_grid.shape)
        I[i] = z_grid

    plt.imshow(I.sum(axis=0),extent=(-limit, limit, -limit, limit))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Likelihood')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return None