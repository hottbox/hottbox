"""
This module contains useful functions for the iterative tensor routines
and data type conversions
"""
import numpy as np
from functools import reduce
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT


def sliceT(tensor, inds, mode, overwrite=None):
    """ Equivalent to multidimnesional slicing

    Parameters
    ----------
    tensor : Tensor
        Tensor to slice at position
    inds : int
        The index of the axis. e.g [:,:,0] will be at mode=2, inds=0
    mode : int
        The axis to access
    overwrite : Tensor
        Overwrite slice with a subtensor
    Returns
    -------
        Numpy function for creating arrays
    """
    tensl = np.array([slice(None)] * tensor.ndim)
    tensl[mode] = inds
    tensl = tensl.tolist()
    if overwrite is not None:
        tensor[tensl] = overwrite
    return tensor[tensl]


