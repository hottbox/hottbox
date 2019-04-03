import numpy as np
from functools import reduce
from ...core.structures import Tensor, TensorCPD, TensorTKD, TensorTT


def toeplitz_matrix(r, c=None):
    r = np.asarray(r).flatten()
    if c is None:
        c = r.conjugate()
    else:
        c = np.asarray(c).flatten()

    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    place = a + b
    # place contains the positional indicies such that 
    # that vals[place] would be a Toeplitz matrix.
    return vals[place]


def hankel_matrix(r, c=None):
    r = r[::-1]
    if c is not None:
        c = c[::-1]
    return toeplitz_matrix(r, c)[::-1]
