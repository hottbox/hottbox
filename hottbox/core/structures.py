"""
Classes for different tensor representations
"""

import numpy as np
from .operations import unfold, fold, mode_n_product


class Tensor(object):
    """ This class describes multidimensional data.

    All its methods implement all common operation on a tensor alone

    Parameters
    ----------
    data : np.ndarray
        N-dimensional array
    _orig_shape : tuple
        Original shape of a tensor. Defined at the object creation for convenience during unfolding and folding.
        Can potentially cause a lot of problems in a future.
    """
    # TODO: add description for the tensor and the factor matrices/modes etc. (Through pandas integration???)
    # TODO: implement unfolding and folding to tensors of an arbitrary order

    def __init__(self, array) -> None:
        """

        Parameters
        ----------
        array : {np.ndarray, Tensor}
        """
        # TODO: covert data to a specific data type (int, float etc)
        if isinstance(array, Tensor):
            self.data = array.data
            self._orig_shape = array._orig_shape
        else:
            self.data = array
            self._orig_shape = array.shape

    def copy(self):
        """ Produces a copy of itself

        Returns
        -------
        Tensor
        """
        return Tensor(self)

    @property
    def frob_norm(self):
        """ Frobenious norm of a tensor

        Returns
        -------
        float
        """
        # return np.sqrt(np.sum(self.data ** 2))
        return np.linalg.norm(self.data)

    @property
    def shape(self):
        """ Sizes of all dimensions of a tensor

        Returns
        -------
        tuple
        """
        return self.data.shape

    @property
    def order(self):
        """ Order of a tensor

        Returns
        -------
        int
        """
        return self.data.ndim

    @property
    def size(self):
        """ Number of elements in a tensor

        Returns
        -------
        int
        """
        return self.data.size

    def unfold(self, mode, inplace=True):
        """ Perform mode-n unfolding to a matrix

        Parameters
        ----------
        mode : int
            Specifies a mode along which a `tensor` will be unfolded
        inplace : bool
            If True, then modifies itself.
            If False, then creates new object (copy)

        Returns
        ----------
        tensor : Tensor
            Unfolded version of a tensor
        """
        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        tensor.data = unfold(self.data, mode)
        return tensor

    def fold(self, inplace=True):
        """ Fold to the original shape (undo self.unfold)

        Parameters
        ----------
        inplace : bool
            If True, then modifies itself.
            If False, then creates new object (copy)

        Returns
        ----------
        tensor : Tensor
            Tensor of original shape (self._orig_shape)
        """
        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        mode = tensor._orig_shape.index(tensor.shape[0])
        tensor.data = fold(self.data, mode, self._orig_shape)
        return tensor

    def mode_n_product(self, matrix, mode, inplace=True):
        """ Mode-n product of a tensor with a matrix

        Parameters
        ----------
        matrix : {Tensor, np.ndarray}
            2D array
        mode : int
            Specifies mode along which a tensor is multiplied by a `matrix`
        inplace : bool
            If True, then modifies itself.
            If False, then creates new object (copy)

        Returns
        -------
        tensor : Tensor
            The result of the mode-n product of a tensor with a `matrix` along specified `mode`.

        Notes
        -------
        Remember that mode_n product changes the shape of the tensor. Presumably, it also changes the interpretation
        of that mode
        """
        # TODO: Think about the way to change mode_description
        if isinstance(matrix, np.ndarray):
            matrix = Tensor(matrix)
        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        tensor.data = mode_n_product(tensor=tensor.data, matrix=matrix.data, mode=mode)
        tensor._orig_shape = tensor.shape
        return tensor


class BaseTensorTD(object):
    """
    This class provides a general interface for a tensor represented through a tensor decomposition.
    """
    def __init__(self):
        pass

    @property
    def order(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def rank(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def size(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def frob_norm(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def unfold(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def fold(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def mode_n_product(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def reconstruct(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')


class TensorCPD(BaseTensorTD):
    """ Representation of a tensor in the CPD form.

    Parameters
    ----------
    fmat : list[np.ndarray]
        List of factor matrices for the CP representation of a tensor
    core_values : np.ndarray
        Array of coefficients on the super-diagonal of a core for the CP representation of a tensor
    """
    def __init__(self, fmat, core_values):
        super(TensorCPD, self).__init__()
        self.fmat = fmat
        self.core_values = core_values

    @property
    def order(self):
        """ Order of a tensor represented through the CPD

        Returns
        -------
        order : int
        """
        order = len(self.fmat)
        return order

    @property
    def rank(self):
        """ Rank of the CP representation of a tensor.

        Returns
        -------
        rank : tuple

        Notes
        -----
        Most often referred to as the Kryskal rank
        """
        fmat = self.fmat[0]
        rank = (fmat.shape[1],)
        return rank

    @property
    def core(self):
        """ Core tensor of the CP representation of a tensor

        Returns
        -------
        core_tensor : Tensor
        """
        core_tensor = super_diag_tensor(self.order, values=self.core_values)
        return core_tensor

    @property
    def reconstruct(self):
        """ Converts the CP representation of a tensor into a full tensor

        Returns
        -------
        tensor : Tensor
        """
        tensor = self.core
        for mode, fmat in enumerate(self.fmat):
            tensor.mode_n_product(fmat, mode=mode, inplace=True)
        return tensor


class TensorTKD(BaseTensorTD):
    """ Representation of a tensor in the Tucker form.

    Parameters
    ----------
    fmat : list[np.ndarray]
        List of factor matrices for the Tucker representation of a tensor
    core : Tensor
        Core of the Tucker representation of a tensor
    """
    def __init__(self, fmat, core):
        super(TensorTKD, self).__init__()
        self.fmat = fmat
        self.core = core

    @property
    def order(self):
        """ Order of a tensor represented through the TKD

        Returns
        -------
        order : int
        """
        order = len(self.fmat)
        return order

    @property
    def rank(self):
        """ Multi-linear rank of the Tucker representation of a tensor

        Returns
        -------
        rank : tuple

        Notes
        -----
        Most often referred to as the Tucker rank
        """
        rank = tuple([fmat.shape[1] for fmat in self.fmat])
        return rank

    @property
    def reconstruct(self):
        """ Converts the Tucker representation of a tensor into a full tensor

        Returns
        -------
        tensor : Tensor
        """
        tensor = self.core
        for mode, fmat in enumerate(self.fmat):
            tensor.mode_n_product(fmat, mode=mode, inplace=True)
        return tensor


class TensorTT(BaseTensorTD):
    """ Representation of a tensor in the TT form.

    Parameters
    ----------
    cores : list[Tensor]
        List of core Tensors for the Tensor Train representation of a tensor.
    full_shape : tuple
        Shape of the full tensor (``TensorTT.reconstruct.shape``). Makes the reconstruction process easier.
    """
    def __init__(self, cores, full_shape):
        super(TensorTT, self).__init__()
        self.cores = cores
        self.full_shape = full_shape

    @property
    def order(self):
        """ Order of a tensor represented through the TT

        Returns
        -------
        order : int
        """
        return len(self.cores)

    @property
    def rank(self):
        """ Rank of the TT representation of a tensor

        Returns
        -------
        rank : tuple

        Notes
        -----
        Most often referred to as the TT rank
        """
        return tuple([core.shape[-1] for core in self.cores[:-1]])

    def reconstruct(self):
        """ Converts the TT representation of a tensor into a full tensor

        Returns
        -------
        tensor : Tensor
        """
        rank = self.rank + (1,)
        data = self.cores[0]
        for i, core in enumerate(self.cores[1:]):
            shape_2d = [rank[i],rank[i+1]*self.full_shape[i+1]]
            core_flat = np.reshape(core.data, shape_2d, order='F')
            data = np.reshape(data, [-1, rank[i]], order='F')
            data = np.dot(data, core_flat)
        data = np.reshape(data, self.full_shape, order='F')
        tensor = Tensor(data)
        return tensor


def super_diag_tensor(order, values=None):
    """ Super-diagonal tensor of the specified `order`.

    Parameters
    ----------
    order : int
        Desired order of the tensor
    values : np.ndarray
        Array of values on the super-diagonal of a tensor. By default contains only ones.
        Length of this vector defines the Kryskal rank of a tensor.

    Returns
    -------
    tensor : Tensor
    """
    rank = values.size
    shape = (rank,) * order
    if values is None:
        values = np.ones(rank)
    core = np.zeros(shape)
    core[np.diag_indices(rank, ndim=order)] = values
    tensor = Tensor(core)
    return tensor


def residual_tensor(tensor_A, tensor_B):
    """ Residual tensor

    Parameters
    ----------
    tensor_A : Tensor
    tensor_B : {Tensor, TensorCPD, TensorTKD}

    Returns
    -------
    residual : Tensor
    """
    if isinstance(tensor_B, TensorCPD) or isinstance(tensor_B, TensorTKD):
        residual = Tensor(tensor_A.data - tensor_B.reconstruct.data)
    elif isinstance(tensor_B, Tensor):
        residual = Tensor(tensor_A.data - tensor_B.data)
    else:
        raise TypeError('Unknown data type of the approximation')
    return residual