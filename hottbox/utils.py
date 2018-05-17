"""
This module contains useful functions for the iterative tensor routines
and data type conversions
"""
import numpy as np
from functools import reduce
from .core.structures import Tensor


def quick_tensor(shape, base='arange'):
    """ Create quick tensor

    Parameters
    ----------
    shape : tuple
        Desired shape of the tensor
    base : str
        Base function that generate values for the tensor
    Returns
    -------
    tensor : Tensor
    """
    base_function = {'arange': np.arange,
                     'randn': np.random.randn,
                     'rand': np.random.rand,
                     'ones': np.ones}.get(base, np.arange)
    size = reduce(lambda x, y: x * y, shape)
    array = base_function(size).reshape(shape)
    tensor = Tensor(array=array)
    return tensor
