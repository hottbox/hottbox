import numpy as np
from ...core.structures import Tensor
from ...utils.gen.matrices import genToeplitzMatrix
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
    shape : int
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
        print("not yet implemented")
    return Tensor(array=tensor)

def sparse(shape, distr='uniform', distr_type=0, fxdind=None, pct=0.05):
    """ Generates a dense or sparse tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    shape : int
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
        tensor[indx] = _predefined_distr(distr,sz)
        tensor = tensor.reshape(shape)
    else:
        print("not yet implemented")

    return Tensor(array=tensor)

def superdiagonal(shape, distr='uniform', dataset=[None]):
    """ Generates a tensor of any dimension with random or specified numbers accross the superdiagonal and zeros elsewhere
    
    Parameters
    ----------
    shape : int
        specifies the dimensions of the tensor
    distr (optional): string
        Specifies the random generation using a class of the numpy.random module
    values : List
        User defined number for the super diagonal
    
    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """

    if shape[1:] != shape[:-1]:
        print("Must have equal dimensions "
              + "for a supersymmetric matrix")
    inds = shape[0]
    tensor = np.zeros(shape)
    
    dataset = np.asarray(dataset).flatten()
    if dataset == [None]:
        dataset = _predefined_distr(distr, inds)
    if len(dataset) != inds:
        print("The specified values do not match"\
                +"the shape of the tensor provided")
    np.fill_diagonal(tensor, dataset)
    return Tensor(array=tensor)

"""def supersymmetric(shape):"""
