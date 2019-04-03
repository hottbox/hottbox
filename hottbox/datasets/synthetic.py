import numpy as np
from ..core.structures import Tensor


def _predefined_distr(distr, shape):
    """

    Parameters
    ----------
    distr :
    shape :

    Returns
    -------

    """
    distrlist = {'uniform': np.random.uniform(size=shape),
                 'normal': np.random.normal(size=shape),
                 'triangular': np.random.triangular(-1, 0, 1, size=shape),
                 'standard-t': np.random.standard_t(10, size=shape),
                 'ones': np.ones(shape),
                 'zeros': np.zeros(shape)}
    if distr not in distrlist:
        raise NameError("The distribution {} is not an available one.\
         Please refer to the list of implementations: {}".format(distr, distrlist.keys()))
    return distrlist[distr]


def make_clusters(dims, centers=3, n_samples=1000, center_bounds=(-10.0, 10.0), std=0.5, return_centers=False):
    """ Generates a tensor of any dimension with isotropic gaussian blobs as clusters

    Parameters
    ----------
    shape : tuple(int)
        specifies the dimensions of the tensor
    n_samples : int or list(int)
        Specifies the size of each clusters
    centers (optional) : int or list(tuples)
        The number of clusters in the dataset and their size (can be a list)
    center_bounds (optional) : tuple(float, float)
        Specifies the bound (min, max) for generating the centers

    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """ 

    tensor = np.array([]).reshape(0, 1, dims)
    
    if isinstance(centers, int):
        centroids = np.random.uniform(*center_bounds, size=(centers, 1, dims))
    else:
        centroids = centers 
        
    n_cent = len(centroids)
    
    if isinstance(n_samples, int):
        n_samples = [n_samples//n_cent]*n_cent

    if len(n_samples) != n_cent:
        raise ValueError("The number of samples specified do not match the number" +
                         "of centers")

    for s_size, center in zip(n_samples, centroids):
        cl = np.random.normal(loc=center, scale=std, size=(s_size, 1, dims))
        tensor = np.concatenate((tensor, cl))

    tensor = np.asarray(tensor)
    if return_centers:
        return Tensor(array=tensor), centroids 
    else:
        return Tensor(array=tensor)
