import numpy as np
from functools import reduce
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
import itertools

def is_toep_matrix(mat):
    """ Utility for checking if a matrix is a Toeplitz matrix
    Parameters
    ----------
    mat : np.ndarray
        n x m array
    Returns
    -------
        Boolean indicating if Toeplitz matrix
    """
    n,m = mat.shape
    # Horizontal diagonals
    for off in range(1,m):
        if np.ptp(np.diagonal(mat, offset=off)):
            return False
    # Vertical diagonals
    for off in range(1,n):
        if np.ptp(np.diagonal(mat, offset=-off)):
            return False
    # we only reach here when all elements 
    # in given diagonal are same 
    return True

# Currently recursive, TODO: improve efficiency
def is_toep_tensor(tensor, modes=[0,1]):
    """ Utility for checking if a Tensor is a Toeplitz Tensor
    Parameters
    ----------
    mat : np.ndarray
        n x m array
    Returns
    -------
        Boolean indicating if Toeplitz matrix
    """
    if tensor.order <= 2:
        return is_toep_matrix(tensor.data)
    sz = np.asarray(tensor.shape)
    availmodes = np.setdiff1d(np.arange(len(sz)),modes)
    for idx, mode in enumerate(availmodes):
        dim = sz[mode]
        # Go through each dim
        for i in range(dim):
            t = tensor.access(i,mode)
            t = Tensor(t)
            if not(is_toep_tensor(t)): 
                print ("Wrong slice: \n{}\n{}".format(t,(i,idx)))
                return False
    return True


def is_super_symmetric(tensor):
    tensor = tensor.data
    idx = np.arange(len(tensor.shape))
    inds = itertools.permutations(idx)
    for k,i in enumerate(inds):
        s = np.transpose(tensor,np.array(i))
        if not np.allclose(tensor,s, atol=1e-4, equal_nan=True):
            print("{} \n is not the same as \n {}".format(tensor, s))
            return False
        
    return True
