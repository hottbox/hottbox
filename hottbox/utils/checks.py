import numpy as np
from functools import reduce
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from .utils import sliceT

def isToepMatrix(mat):
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
def isToepTensor(tensor, modes=[0,1]):
    """ Utility for checking if a Tensor is a Toeplitz Tensor
    Parameters
    ----------
    mat : np.ndarray
        n x m array
    Returns
    -------
        Boolean indicating if Toeplitz matrix
    """
    tensor = tensor.data
    if tensor.ndim <= 2:
        return isToepMatrix(tensor)
    sz = np.asarray(tensor.shape)
    availmodes = np.setdiff1d(np.arange(len(sz)),modes)
    for idx, mode in enumerate(availmodes):
        dim = sz[mode]
        # Go through each dim
        for i in range(dim):
            t = sliceT(tensor,i,mode)
            if not(isToepTensor(t)): 
                print ("Wrong slice: \n{}\n{}".format(t,(i,idx)))
                return False
    return True

