import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


# def unitTest(distribution,limits=[-1,1],n_points = 1000):
#     """Integrate a density over a given limit in 2d-space
#     To make a unit-test and verify that it is a density
#     Should return value close to 1.0

#     Inputs
#         distribution        : A tfd distribution in 2d
#         limits (list)       : Either a list with limits [-X,X] or scalar: X
#         n_points (int)      : Number of point to integrate. Increate for precision

#     Outputs
#         integrand (float)   : Integration over area (should be 1)
#     """

#     # Construct meshgrid
#     if isinstance(limits,list):
#         x,dx = np.linspace(limits[0],limits[1],n_points,retstep=True)
#         y,dy = np.linspace(limits[0],limits[1],n_points,retstep=True)
#     else:
#         x,dx = np.linspace(-limits,limits,n_points,retstep=True)
#         y,dy = np.linspace(-limits,limits,n_points,retstep=True)
#     x_grid, y_grid = np.meshgrid(x, y)
#     X = np.array([x_grid.ravel(), y_grid.ravel()]).T
    
#     # Get density
#     p = distribution.prob(X).numpy()
    
#     # Compute integrand
#     integrand = np.sum(p)*dx*dy
#     return integrand
def unitTest(model,limits=[-1,1],n_points = 1000):
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
    if isinstance(limits,list):
        x,dx = np.linspace(limits[0],limits[1],n_points,retstep=True)
        y,dy = np.linspace(limits[0],limits[1],n_points,retstep=True)
    else:
        x,dx = np.linspace(-limits,limits,n_points,retstep=True)
        y,dy = np.linspace(-limits,limits,n_points,retstep=True)
    x_grid, y_grid = np.meshgrid(x, y)
    X = np.array([x_grid.ravel(), y_grid.ravel()]).T
    
    # Get density
    p_log = model(X).numpy()
    p = np.exp(p_log)
    
    # Compute integrand
    integrand = np.sum(p)*dx*dy
    return integrand

#%% Function examples:

if __name__ == '__main__':
    
    # Initialize a distribution
    dist = tfd.MultivariateNormalDiag(loc=[0,0],scale_diag=[1,1])

    # Compute unit test over a good inverval
    integrand = unitTest(dist,limits=4)
    
    print(f'Integrand is {np.round(integrand,4)}')
