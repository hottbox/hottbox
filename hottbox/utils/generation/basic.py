import itertools
import numpy as np
from hottbox.core.structures import Tensor, BaseTensorTD


def _predefined_distr(distr, shape):
    """

    Parameters
    ----------
    distr : str
    shape : tuple

    Returns
    -------
    np.ndarray
    """
    distrlist = {'uniform': np.random.uniform(size=shape),
                 'normal': np.random.normal(size=shape),
                 'triangular': np.random.triangular(-1, 0, 1, size=shape),
                 'standard-t': np.random.standard_t(10, size=shape),
                 'ones': np.ones(shape),
                 'zeros': np.zeros(shape)}
    if distr not in distrlist:
        raise NameError("The distribution {} is not an available one. "
                        "Please refer to the list of implementations: {}".format(distr, distrlist.keys()))
    return distrlist[distr]


def dense_tensor(shape, distr='uniform', distr_type=0, fxdind=None):
    """ Generates a dense tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    shape : tuple
        Specifies the dimensions of the tensor
    distr : str, optional
        Specifies the random generation using a class of the numpy.random module
    distr_type : int, optional
        Number of indices to not fix. 0 will be applied globally, 1 will apply to fibers, 2 to slices, etc.
    
    Returns
    -------
    Tensor
        Generated tensor according to the parameters specified
    """

    # fxdind: fixed indices
    if distr_type == 0:
        data = _predefined_distr(distr, shape)
    else:
        data = np.random.uniform(size=shape)
        raise NotImplementedError('Not implemented in dataset (basic) class')
    return Tensor(array=data)


def sparse_tensor(shape, distr='uniform', distr_type=0, fxdind=None, pct=0.05):
    """ Generates a sparse tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    shape : tuple
        Specifies the dimensions of the tensor
    distr : str, optional
        Specifies the random generation using a class of the numpy.random module
    distr_type : int, optional
        Number of indices to not fix. 0 will be applied globally, 1 will apply to fibers, 2 to slices, etc.
    pct : float, optional
        Percentage of the dataset to be filled
    
    Returns
    -------
    Tensor
        Generated tensor according to the parameters specified
    """
    data_size = np.product(shape)
    if distr_type == 0:
        number_non_zero_values = int(data_size * pct)
        data = np.zeros(data_size)
        index = np.random.randint(low=0, high=data_size, size=number_non_zero_values)
        data[index] = _predefined_distr(distr, number_non_zero_values)
        data = data.reshape(shape)
    else:
        raise NotImplementedError('Not implemented in dataset (basic) class')

    return Tensor(array=data)

def super_diagonal_tensor(shape, distr='ones', values=None):
    """ Generates a tensor of any dimension with random or specified numbers across the super-diagonal and zeros elsewhere
    
    Parameters
    ----------
    shape : tuple
        Specifies the dimensions of the tensor
        ``len(shape)`` defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    distr : str, optional
        Specifies the random generation using a class of the numpy.random module
    values : list
        Array of values on the super-diagonal of a tensor

    Returns
    -------
    Tensor
        Generated tensor according to the parameters specified
    """
    if not isinstance(shape, tuple):
        raise TypeError("Parameter `shape` should be passed as a tuple!")
    
    if shape[1:] != shape[:-1]:
            raise ValueError("All values in `shape` should have the same value!")
    inds = shape[0]
    data = np.zeros(shape)
    
    if values is None:
        values = _predefined_distr(distr, inds)
    if len(values) != inds:
        raise ValueError("Dimension mismatch! The specified values do not match "
                         "the specified shape of the tensor provided ({} != {})".format(len(values), inds))
    values = np.asarray(values).flatten()
    np.fill_diagonal(data, values)
    return Tensor(array=data)


def super_diag_tensor(shape, values=None):
    """ Super-diagonal tensor of the specified `order`.

    Parameters
    ----------
    shape : tuple
        Desired shape of the tensor.
        ``len(shape)`` defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    values : np.ndarray
        Array of values on the super-diagonal of a tensor. By default contains only ones.
        Length of this vector defines Kryskal rank which is equal to ``shape[0]``.

    Returns
    -------
    tensor : Tensor
    """
    order = len(shape)
    rank = shape[0]

    if not isinstance(shape, tuple):
        raise TypeError("Parameter `shape` should be passed as a tuple!")
    if not all(mode_size == shape[0] for mode_size in shape):
        raise ValueError("All values in `shape` should have the same value!")

    if values is None:
        values = np.ones(rank)  # set default values
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("The `values` should be 1-dimensional numpy array!")
        if values.size != rank:
            raise ValueError("Dimension mismatch! Not enough or too many `values` for the specified `shape`:\n"
                             "{} != {} (values.size != shape[0])".format(values.size, rank))
    else:
        raise TypeError("The `values` should be passed as a numpy array!")

    core = np.zeros(shape)
    core[np.diag_indices(rank, ndim=order)] = values
    tensor = Tensor(core)
    return tensor


def super_symmetric_tensor(shape, tensor=None):
    """ Generates a tensor of equal dimensions with random or specified numbers, with a specified tensor.

    Parameters
    ----------
    shape : tuple
        Specifies the dimensions of the tensor
        ``len(shape)`` defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    tensor : Tensor, optional
        Input tensor to be symmetricised

    Returns
    -------
    Tensor
        Generated tensor according to the parameters specified
    """

    dims = len(shape)
    inds = itertools.permutations(np.arange(dims))
    inds = np.array(list(inds))
    data = np.zeros(shape)
    if tensor is None:
        tensor = dense_tensor(shape)
    for i, _ in enumerate(inds):
        data = data + np.transpose(tensor.data, tuple(inds[i, :]))
    return Tensor(array=data)


def residual_tensor(tensor_orig, tensor_approx):
    """ Residual tensor

    Parameters
    ----------
    tensor_orig : Tensor
    tensor_approx : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    residual : Tensor
    """
    if not isinstance(tensor_orig, Tensor):
        raise TypeError("Unknown data type of original tensor.\n"
                        "The available type for `tensor_A` is `Tensor`")
    # TODO: make use of direct subtraction of tensors
    if isinstance(tensor_approx, Tensor):
        residual = Tensor(tensor_orig.data - tensor_approx.data)
    elif isinstance(tensor_approx, BaseTensorTD):
        residual = Tensor(tensor_orig.data - tensor_approx.reconstruct().data)
    else:
        raise TypeError("Unknown data type of the approximation tensor!\n"
                        "The available types for `tensor_B` are `Tensor`,  `TensorCPD`,  `TensorTKD`,  `TensorTT`")
    return residual
