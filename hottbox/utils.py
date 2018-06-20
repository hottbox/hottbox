"""
This module contains useful functions for the iterative tensor routines
and data type conversions
"""
import numpy as np
from functools import reduce
from .core.structures import Tensor, TensorCPD, TensorTKD, TensorTT


def select_base_function(base):
    """ Util for creating arrays

    Parameters
    ----------
    base : str
        Id of base function
        If not one from {"arange", "randn", "rand", "ones"} then `np.arange` will be used

    Returns
    -------
        Numpy function for creating arrays
    """
    return {"arange": np.arange,
            "randn": np.random.randn,
            "rand": np.random.rand,
            "ones": np.ones}.get(base, np.arange)


def quick_tensor(shape, base="arange"):
    """ Simplified creation of generic tensor

    Parameters
    ----------
    shape : tuple
        Desired shape of the tensor
    base : str
        Id of base function that generates values for the tensor.
        If not one from {"arange", "randn", "rand", "ones"} then `np.arange` will be used.

    Returns
    -------
    tensor : Tensor
    """
    size = reduce(lambda x, y: x * y, shape)
    create_ndarray = select_base_function(base)
    array = np.reshape(create_ndarray(size), shape)
    tensor = Tensor(array=array)
    return tensor


def quick_tensorcpd(full_shape, rank, base="arange"):
    """ Simplified creation of generic tensor in kruskal form

    Parameters
    ----------
    full_shape : tuple
        Desired shape of the tensor when it is reconstructed.
        Values specify the number of rows of the factor matrices.
    rank : tuple
        Desired kruskal rank of the tensor. Specifies the number of columns for all factor matrices
        In order to be consistent with the rest of ``hottbox`` should be in form of `(value,)`
    base : str
        Id of base function that generates values for the components of kruskal tensor.
        If not one from {"arange", "randn", "rand", "ones"} then `np.arange` will be used.

    Returns
    -------
    tensor_cpd : TensorCPD
    """
    create_ndarray = select_base_function(base)
    fmat_shapes = zip(full_shape, [rank[0] for _ in range(len(full_shape))])
    core_values = np.ones(rank[0])
    fmat = []
    for fmat_shape in fmat_shapes:
        size = reduce(lambda x, y: x * y, fmat_shape)
        fmat.append(np.reshape(create_ndarray(size), fmat_shape))
    tensor_cpd = TensorCPD(fmat=fmat, core_values=core_values)
    return tensor_cpd


def quick_tensortkd(full_shape, rank, base="arange"):
    """ Simplified creation of generic tensor in tucker form

    Parameters
    ----------
    full_shape : tuple
        Desired shape of the tensor when it is reconstructed.
        Values specify the number of rows of the factor matrices.
    rank : tuple
        Desired multi-linear rank of the tensor. Specifies the number of columns for all factor matrices.
        Should be of the same length as parameter `shape`
    base : str
        Id of base function that generates values for the components of tucker tensor.
        If not one from {"arange", "randn", "rand", "ones"} then `np.arange` will be used.

    Returns
    -------
    tensor_tkd : TensorTKD
    """
    create_ndarray = select_base_function(base)
    fmat_shapes = zip(full_shape, rank)
    core_values = np.ones(rank)
    fmat = []
    for shape in fmat_shapes:
        size = reduce(lambda x, y: x * y, shape)
        fmat.append(np.reshape(create_ndarray(size), shape))
    tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)
    return tensor_tkd


def quick_tensortt(full_shape, rank, base="arange"):
    """ Simplified creation of generic tensor in tensor train form

    Parameters
    ----------
    full_shape : tuple
        Desired shape of the tensor when it is reconstructed.
    rank : tuple
        Desired tt rank of the tensor.
    base : str
        Id of base function that generates values for the components of tensor train tensor.
        If not one from {"arange", "randn", "rand", "ones"} then `np.arange` will be used.

    Returns
    -------
    tensor_tt : TensorTT
    """
    create_ndarray = select_base_function(base)
    tt_ranks_l = rank[:-1]
    tt_ranks_r = rank[1:]
    number_of_middle_cores = len(tt_ranks_l)
    first_core_shape = (full_shape[0], rank[0])
    last_core_shape = (rank[-1], full_shape[-1])
    middle_cores_shape = [(tt_ranks_l[i], full_shape[i + 1], tt_ranks_r[i]) for i in range(number_of_middle_cores)]
    core_shapes = [first_core_shape] + middle_cores_shape + [last_core_shape]
    core_values = []
    for shape in core_shapes:
        size = reduce(lambda x, y: x * y, shape)
        core_values.append(np.reshape(create_ndarray(size), shape))
    tensor_tt = TensorTT(core_values=core_values, ft_shape=full_shape)
    return tensor_tt
