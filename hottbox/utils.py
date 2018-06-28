"""
This module contains useful functions for the iterative tensor routines
and data type conversions
"""
import numpy as np
from functools import reduce
from .core.structures import Tensor, TensorCPD, TensorTKD, TensorTT


def _select_base_function(base):
    """ Utility for creating arrays

    Parameters
    ----------
    base : str
        Id of base function.
        If not one from ``{"arange", "randn", "rand", "ones"}`` then `np.arange` will be used.

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
        If not one from ``{"arange", "randn", "rand", "ones"}`` then ``np.arange`` will be used.

    Returns
    -------
    tensor : Tensor

    Examples
    --------
        >>> from hottbox.utils import quick_tensor
        >>> tensor = quick_tensor(shape=(2, 3, 4))
        >>> print(tensor)
            This tensor is of order 3 and consists of 24 elements.
            Sizes and names of its modes are (2, 3, 4) and ['mode-0', 'mode-1', 'mode-2'] respectively.
        >>> print(tensor.data)
            [[[ 0  1  2  3]
              [ 4  5  6  7]
              [ 8  9 10 11]]
             [[12 13 14 15]
              [16 17 18 19]
              [20 21 22 23]]]
        >>> tensor = quick_tensor(shape=(2, 3, 4), base="ones")
        >>> print(tensor)
            This tensor is of order 3 and consists of 24 elements.
            Sizes and names of its modes are (2, 3, 4) and ['mode-0', 'mode-1', 'mode-2'] respectively.
        >>> print(tensor.data)
            [[[1. 1. 1. 1.]
              [1. 1. 1. 1.]
              [1. 1. 1. 1.]]
             [[1. 1. 1. 1.]
              [1. 1. 1. 1.]
              [1. 1. 1. 1.]]]
    """
    size = reduce(lambda x, y: x * y, shape)
    create_ndarray = _select_base_function(base)
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
        Desired kruskal rank of the tensor. Specifies the number of columns for all factor matrices.
        In order to be consistent with the rest of ``hottbox`` should be in form of ``(value,)``
    base : str
        Id of base function that generates values for the components of kruskal tensor.
        If not one from ``{"arange", "randn", "rand", "ones"}`` then ``np.arange`` will be used.

    Returns
    -------
    tensor_cpd : TensorCPD

    Examples
    --------
        >>> from hottbox.utils import quick_tensorcpd
        >>> tensor_cpd = quick_tensorcpd(full_shape=(3, 4, 5), rank=(2,), base="ones")
        >>> print(tensor_cpd)
            Kruskal representation of a tensor with rank=(2,).
            Factor matrices represent properties: ['mode-0', 'mode-1', 'mode-2']
            With corresponding latent components described by (3, 4, 5) features respectively.
        >>> tensor = tensor_cpd.reconstruct()
        >>> print(tensor)
            This tensor is of order 3 and consists of 60 elements.
            Sizes and names of its modes are (3, 4, 5) and ['mode-0', 'mode-1', 'mode-2'] respectively.
    """
    create_ndarray = _select_base_function(base)
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
        Should be of the same length as parameter ``full_shape``
    base : str
        Id of base function that generates values for the components of tucker tensor.
        If not one from ``{"arange", "randn", "rand", "ones"}`` then ``np.arange`` will be used.

    Returns
    -------
    tensor_tkd : TensorTKD

    Examples
    --------
        >>> from hottbox.utils import quick_tensortkd
        >>> tensor_tkd = quick_tensortkd(full_shape=(5, 6, 7), rank=(2, 3, 4), base="ones")
        >>> print(tensor_tkd)
            Tucker representation of a tensor with multi-linear rank=(2,).
            Factor matrices represent properties: ['mode-0', 'mode-1', 'mode-2']
            With corresponding latent components described by (5, 6, 7) features respectively.
        >>> tensor = tensor_tkd.reconstruct()
        >>> print(tensor)
            This tensor is of order 3 and consists of 210 elements.
            Sizes and names of its modes are (5, 6, 7) and ['mode-0', 'mode-1', 'mode-2'] respectively.
    """
    create_ndarray = _select_base_function(base)
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
        If not one from ``{"arange", "randn", "rand", "ones"}`` then ``np.arange`` will be used.

    Returns
    -------
    tensor_tt : TensorTT

    Examples
    --------
        >>> from hottbox.utils import quick_tensortt
        >>> tensor_tt = quick_tensortt(full_shape=(3, 4, 5), rank=(2, 3), base="ones")
        >>> print(tensor_tt)
            Tensor train representation of a tensor with tt-rank=(2, 3).
            Shape of this representation in the full format is (3, 4, 5).
            Physical modes of its cores represent properties: ['mode-0', 'mode-1', 'mode-2']
        >>> tensor = tensor_tt.reconstruct()
        >>> print(tensor)
            This tensor is of order 3 and consists of 60 elements.
            Sizes and names of its modes are (3, 4, 5) and ['mode-0', 'mode-1', 'mode-2'] respectively.
    """
    create_ndarray = _select_base_function(base)
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
    tensor_tt = TensorTT(core_values=core_values)
    return tensor_tt
