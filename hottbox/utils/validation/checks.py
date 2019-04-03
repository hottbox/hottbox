import numpy as np
from hottbox.core.structures import Tensor
import itertools


def is_toeplitz_matrix(mat):
    """ Checks if ``matrix`` has a Toeplitz structure

    Parameters
    ----------
    mat : np.ndarray
        Input array to check

    Returns
    -------
        Boolean indicating if Toeplitz matrix
    """
    n, m = mat.shape
    # Horizontal diagonals
    for off in range(1, m):
        if np.ptp(np.diagonal(mat, offset=off)):
            return False
    # Vertical diagonals
    for off in range(1, n):
        if np.ptp(np.diagonal(mat, offset=-off)):
            return False
    # we only reach here when all elements 
    # in given diagonal are same 
    return True


# Currently recursive, TODO: improve efficiency
def is_toeplitz_tensor(tensor, modes=None):
    """ Checks if ``tensor`` has Toeplitz structure

    Parameters
    ----------
    tensor : Tensor
        Input tensor to check

    Returns
    -------
        Boolean indicating if Toeplitz matrix
    """
    if tensor.order <= 2:
        return is_toeplitz_matrix(tensor.data)
    if modes is None:
        modes = [0, 1]
    sz = np.asarray(tensor.shape)
    availmodes = np.setdiff1d(np.arange(len(sz)), modes)
    for idx, mode in enumerate(availmodes):
        dim = sz[mode]
        #  Go through each dim
        for i in range(dim):
            t = tensor.access(i, mode)
            t = Tensor(t)
            if not(is_toeplitz_tensor(t)):
                print("Wrong slice: \n{}\n{}".format(t, (i, idx)))
                return False
    return True


def is_super_symmetric(tensor):
    """ Checks if ``tensor`` has supers-symmetric structure

    Parameters
    ----------
    tensor : Tensor
        Input tensor to check

    Returns
    -------
        Boolean indicating if super-symmetric tensor
    """
    tensor = tensor.data
    idx = np.arange(len(tensor.shape))
    inds = itertools.permutations(idx)
    for i in inds:
        s = np.transpose(tensor, np.array(i))
        if not np.allclose(tensor, s, atol=1e-4, equal_nan=True):
            print("{} \n is not the same as \n {}".format(tensor, s))
            return False
        
    return True
