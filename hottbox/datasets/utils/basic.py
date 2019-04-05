import numpy as np
from ...core.structures import Tensor
import itertools


def _predefined_distr(distr, shape):
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


def dense(shape, distr='uniform', distr_type=0, fxdind=None):
    """ Generates a dense or sparse tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    shape : tuple(int)
        specifies the dimensions of the tensor
    distr (optional): string
        Specifies the random generation using a class of the numpy.random module
    distr_type (optional) : int
        Number of indices to not fix. 0 will be applied globally, 1 will apply to fibers, 2 to slices, etc.
    
    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """

    # fxdind: fixed indices
    if distr_type == 0:
        tensor = _predefined_distr(distr, shape)
    else:
        tensor = np.random.uniform(size=shape)
        raise NotImplementedError('Not implemented in dataset (basic) class')
    return Tensor(array=tensor)


def sparse(shape, distr='uniform', distr_type=0, fxdind=None, pct=0.05):
    """ Generates a dense or sparse tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    shape : tuple(int)
        specifies the dimensions of the tensor
    distr (optional): string
        Specifies the random generation using a class of the numpy.random module
    distr_type (optional) : int
        Number of indices to not fix. 0 will be applied globally, 1 will apply to fibers, 2 to slices, etc.
    pct (optional) : float
        Percentage of the dataset to be filled
    
    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """

    tensorsz = np.product(shape)
    if distr_type == 0:
        sz = int(tensorsz * pct)
        tensor = np.zeros(tensorsz)
        indx = np.random.randint(low=0, high=tensorsz, size=sz)
        tensor[indx] = _predefined_distr(distr, sz)
        tensor = tensor.reshape(shape)
    else:
        raise NotImplementedError('Not implemented in dataset (basic) class')

    return Tensor(array=tensor)


def superdiagonal(shape, distr='uniform', values=[None]):
    """ Generates a tensor of any dimension with random or specified numbers accross the superdiagonal and zeros elsewhere
    
    Parameters
    ----------
    shape : tuple(int)
        specifies the dimensions of the tensor
        ``len(shape)`` defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    distr (optional): string
        Specifies the random generation using a class of the numpy.random module
    values : List
        Array of values on the super-diagonal of a tensor

    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """
    if not isinstance(shape, tuple):
        raise TypeError("Parameter `shape` should be passed as a tuple!")
    
    if shape[1:] != shape[:-1]:
            raise ValueError("All values in `shape` should have the same value!")
    inds = shape[0]
    tensor = np.zeros(shape)
    
    if values == [None]:
        values = _predefined_distr(distr, inds)
    if len(values) != inds:
        raise ValueError("Dimension mismatch! The specified values do not match "
                         + "the specified shape of the tensor provided ({} != {})".format(len(values), inds))
    values = np.asarray(values).flatten()
    np.fill_diagonal(tensor, values)
    return Tensor(array=tensor)


def supersymmetric(shape, tensor=None):
    """ Generates a tensor of equal dimensions with random or specified numbers, with a specified tensor.

    Parameters
    ----------
    shape : tuple(int)
        specifies the dimensions of the tensor
        ``len(shape)`` defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    tensor (optional): Tensor
        input tensor to be symmetricized

    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """

    dims = len(shape)
    inds = itertools.permutations(np.arange(dims))
    inds = np.array(list(inds))
    A = np.zeros(shape)
    if tensor is None:
        tensor = dense(shape) 
    for i, _ in enumerate(inds):
        A = A + np.transpose(tensor.data, tuple(inds[i,:]))
    return Tensor(array=A)
