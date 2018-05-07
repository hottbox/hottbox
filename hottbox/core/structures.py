"""
Classes for different tensor representations
"""

import numpy as np
from functools import reduce
from collections import OrderedDict
from .operations import unfold, fold, mode_n_product


class Tensor(object):
    """ This class describes multidimensional data.

    All its methods implement all common operation on a tensor alone

    Attributes
    ----------
    _data : np.ndarray
        N-dimensional array
    _ft_shape : tuple
        Shape of a tensor object in normal format (without being in unfolded or folded state)
        Can potentially cause a lot of problems in a future.
    _mode_names : OrderedDict
        Description of the tensor modes
    """

    def __init__(self, array, mode_names=None, ft_shape=None) -> None:
        """ Create object of ``Tensor`` class

        Parameters
        ----------
        array : np.ndarray
            N-dimensional array
        mode_names : OrderedDict
            Description of the tensor modes.
            If nothing is specified then all modes of the created ``Tensor``
            get generic names {0:'mode-0', 1:'mode-1', ...}
        ft_shape : tuple
            Shape of the a tensor in normal format (without being in unfolded or folded state)

        Notes
        -----
            In most cases use the default settings for `ft_shape`, because it affects folding, unfolding etc.
        """
        # TODO: covert data to a specific data type (int, float etc)
        if not isinstance(array, np.ndarray):
            raise TypeError('Input data should be a numpy array')
        self._data = array.copy()
        self._mode_names = self._assign_names(array=array, mode_names=mode_names)
        self._ft_shape = self._assign_ft_shape(array=array, ft_shape=ft_shape)

    def _assign_names(self, array, mode_names):
        """ Generate list of names for the modes of a tensor
        
        Parameters
        ----------
        array : np.ndarray
            N-dimensional array
        mode_names : OrderedDict
            Description of the tensor modes in form of a dictionary where Keys and Values
            correspond to mode number and description respectively

        Returns
        -------
        names : OrderedDict
        """
        if mode_names is None:
            names = OrderedDict([(mode,"mode-{}".format(mode)) for mode in range(array.ndim)])
        else:
            if not isinstance(mode_names, OrderedDict):
                raise TypeError("You should use OrderDict for mode_names!")
            if array.ndim != len(mode_names.keys()):
                raise ValueError("Incorrect number of names for the modes of a tensor: {0} != {1} "
                                 "('array.ndim != len(mode_names.keys())')!\n".format(array.ndim,
                                                                                      len(mode_names.keys())
                                                                                      )
                                 )
            if not all(isinstance(mode, int) for mode in mode_names.keys()):
                raise TypeError("The dict of mode names should contain only integer keys!")
            if not all(mode < array.ndim for mode in mode_names.keys()):
                raise ValueError("All specified modes should not exceed the order of the tensor!")
            if not all(mode >= 0 for mode in mode_names.keys()):
                raise ValueError("All specified values for modes should be non-negative!")

            names = mode_names.copy()
        return names

    def _assign_ft_shape(self, array, ft_shape):
        """ Generate shape for a normal format of a tensor (without being in unfolded or folded state)

        Parameters
        ----------
        array : np.ndarray
            N-dimensional array
        ft_shape : tuple
            Shape for a normal format of a tensor
        Returns
        -------
        shape : tuple
        """
        if ft_shape is None:
            shape = tuple([i for i in array.shape])
        else:
            if not isinstance(ft_shape, tuple):
                raise TypeError("Incorrect type of the parameter `ft_shape`!\n"
                                "It should be tuple")

            size = reduce(lambda x, y: x * y, ft_shape)
            if array.size != size:
                raise ValueError("Values of `ft_shape` are inconsistent with the provided data array ({} != {})!\n"
                                 "reduce(lambda x, y: x * y, ft_shape) != array.size".format(size, array.size))

            shape = tuple([i for i in ft_shape])
        return shape

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : Tensor
            New object of Tensor class with attributes having the same values, but no memory space is shared

        Notes
        -----
            Attribute `_ft_shape` is assigned during object creation based on the shape of data array.
            In order to preserve the original values without sharing memory space, need to redefine them manually.
        """
        array = self.data
        mode_names = self.mode_names
        ft_shape = self.ft_shape
        new_object = Tensor(array=array, mode_names=mode_names, ft_shape=ft_shape)
        return new_object

    @property
    def data(self):
        """ N-dimensional array with data values 
        
        Returns
        -------
        array : np.ndarray
        """
        array = self._data
        return array

    @property
    def ft_shape(self):
        """ Shape of the a tensor in normal format (without being in unfolded or folded state)

        Returns
        -------
        shape : tuple
        """
        shape = self._ft_shape
        return shape

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

    @property
    def mode_names(self):
        """ Description of the tensor modes

        Returns
        -------
        names : OrderedDict
        """
        names = self._mode_names
        return names

    def rename_modes(self, new_mode_names):
        """ Rename modes of a tensor
        
        Parameters
        ----------        
        new_mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Notes
        -----
            `new_mode_name` does not have be OrderedDict since `mode_names` attribute is created at the
            Tensor object creation
        """
        if (len(new_mode_names.keys()) > self.order):
            raise ValueError("Too many mode names have been specified")
        if not all(isinstance(mode, int) for mode in new_mode_names.keys()):
            raise TypeError("The dict of `new_mode_names` should contain only integer keys!")
        if not all(mode < self.order for mode in new_mode_names.keys()):
            raise ValueError("All specified mode values should not exceed the order of the tensor!")
        if not all(mode >= 0 for mode in new_mode_names.keys()):
            raise ValueError("All specified mode keys should be non-negative!")
        self._mode_names.update(new_mode_names)

    def describe(self):
        """ Provides general information about this instance."""
        mode_names = {key:value for key, value in self.mode_names.items()}
        print("This tensor is of order {}, consists of {} elements and its Frobenious norm = {:.2f}.\n"
              "Sizes and names of its modes are {} and {} respectively.".format(self.order, self.size, self.frob_norm,
                                                                                self.shape, mode_names))

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

        Notes
        -----
            Unfolding operation does not change `_ft_shape` attribute
        """
        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        tensor._data = unfold(self.data, mode)

        new_mode_names = {0 : OrderedDict([(mode , tensor.mode_names[mode])]),
                          1 : OrderedDict([(orig_mode,tensor.mode_names[orig_mode]) for orig_mode in tensor.mode_names.keys() if orig_mode != mode])
                          }
        # remove mode names due to collapsed dimensions
        for mode_to_remove in range(2,len(tensor.mode_names.keys())):
            del tensor._mode_names[mode_to_remove]

        tensor.rename_modes(new_mode_names=new_mode_names)
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
            Tensor of original shape (self._ft_shape)

        Notes
        -----
            Folding operation does not change `_ft_shape` attribute
        """
        if inplace:
            tensor = self
        else:
            tensor = self.copy()

        # Do not do anything if the tensor is in the normal form (hadn't been unfolded before)
        if tensor.shape == tensor._ft_shape:
            return tensor

        # --------------- UNFOLD DATA
        # Infer along which mode this instance has been previously unfolded
        mode_0 = list(tensor.mode_names[0].keys())
        mode_1 = list(tensor.mode_names[1].keys())
        folding_mode = mode_0[0]

        # Infer shape of the folded version
        new_shape = [None] * (len(mode_0) + len(mode_1))
        for i in (mode_0 + mode_1):
            new_shape[i] = tensor._ft_shape[i]

        # Update data
        tensor._data = fold(matrix=self.data, mode=folding_mode, shape=new_shape)

        # --------------- UNFOLD DESCRIPTION
        # Create dict with new names
        new_mode_names = {**tensor.mode_names[0], **tensor.mode_names[1]}

        # Sequentially add default description for missing modes in order to accommodate new info
        start = len(tensor.mode_names.keys())
        stop = len(new_mode_names.keys())
        for i in range(start, stop):
            tensor._mode_names[i] = 'mode-{}'.format(i)

        # Update description
        tensor.rename_modes(new_mode_names=new_mode_names)
        return tensor

    def mode_n_product(self, matrix, mode, inplace=True, new_name=None):
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
        new_name : str
            New name for the corresponding `mode` after computing this product.
            See Notes-3 for more info

        Returns
        -------
        tensor : Tensor
            The result of the mode-n product of a tensor with a `matrix` along specified `mode`.

        Notes
        -------
            1. Mode-n product operation changes the `_ft_shape` attribute
            2. Remember that mode-n product changes the shape of the tensor. Presumably, it also changes
               the interpretation of that mode depending on the matrix
            3. If `matrix` is an object of `Tensor` class then you shouldn't specify `new_name`, since
               it will be changed to `matrix.mode_names[0]`
            4. If `matrix.mode_names[0] == "mode-0"` then no changes to `tensor.mode_names` will be made
        """
        # TODO: need to rethink this if statements so it would be easier to follow
        if isinstance(matrix, Tensor) and new_name is not None:
            raise ValueError("Oops... Don't know which name for the mode description to use!\n"
                             "Either use the default value for `new_name=None` or pass numpy array for `matrix.`")
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError("The parameter `new_name` should be of sting type!")

        # Convert to Tensor class, in order to have consistent interface
        if isinstance(matrix, np.ndarray):
            matrix = Tensor(matrix)

        if new_name is None:
            new_name = matrix.mode_names[0]

        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        tensor._data = mode_n_product(tensor=tensor.data, matrix=matrix.data, mode=mode)
        tensor._ft_shape = tensor.shape

        # The only one case when mode name won't be changed
        if new_name != "mode-0":            
            new_mode_names = {mode: new_name}
            tensor.rename_modes(new_mode_names=new_mode_names)

        return tensor


class BaseTensorTD(object):
    """
    This class provides a general interface for a tensor represented through a tensor decomposition.
    """
    def __init__(self):
        pass

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : BaseTensorTD
        """
        cls = self.__class__
        new_object = cls.__new__(cls)
        new_object.__dict__.update(self.__dict__)
        return new_object

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

    Attributes
    ----------
    _fmat : list[np.ndarray]
        Placeholder for a list of factor matrices for the CP representation of a tensor
    _core_values : np.ndarray
        Placeholder for an array of coefficients on the super-diagonal of a core for the CP representation of a tensor
    """
    def __init__(self, fmat, core_values):
        """
        
        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the CP representation of a tensor
        core_values : np.ndarray
            Array of coefficients on the super-diagonal of a core for the CP representation of a tensor
        """
        super(TensorCPD, self).__init__()
        self._fmat = fmat.copy()
        self._core_values = core_values.copy()

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorCPD
        """
        new_object = super(TensorCPD, self).copy()
        return new_object

    @property
    def core(self):
        """ Core tensor of the CP representation of a tensor

        Returns
        -------
        core_tensor : Tensor
        """
        core_tensor = super_diag_tensor(self.order, values=self._core_values)
        return core_tensor

    @property
    def fmat(self):
        """ List of factor matrices for the CP representation of a tensor
        
        Returns
        -------
        factor_matrices : list[np.ndarray]
        """
        factor_matrices = self._fmat
        return factor_matrices

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
    _fmat : list[np.ndarray]
        Placeholder for a list of factor matrices for the Tucker representation of a tensor
    _core_values : np.ndarray
        Placeholder for a core of the Tucker representation of a tensor
    """
    def __init__(self, fmat, core_values):
        """
        
        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the Tucker representation of a tensor
        core_values : np.ndarray
            Core of the Tucker representation of a tensor
        """
        super(TensorTKD, self).__init__()
        self._fmat = fmat.copy()
        self._core_values = core_values.copy()

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorTKD
        """
        new_object = super(TensorTKD, self).copy()
        return new_object

    @property
    def core(self):
        """ Core tensor of the CP representation of a tensor

        Returns
        -------
        core_tensor : Tensor
        """
        core_tensor = Tensor(self._core_values)
        return core_tensor

    @property
    def fmat(self):
        """ List of factor matrices for the Tucker representation of a tensor
        
        Returns
        -------
        factor_matrices : list[np.ndarray]
        """
        factor_matrices = self._fmat
        return factor_matrices

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
    core_values : list[np.ndarray]
        Placeholder for a list of cores for the Tensor Train representation of a tensor.
    full_shape : tuple
        Placeholder for a shape of the full tensor (``TensorTT.reconstruct.shape``). Makes the reconstruction process easier.
    """
    def __init__(self, core_values, full_shape):
        """
        
        Parameters
        ----------
        core_values : list[np.ndarray]
            List of cores for the Tensor Train representation of a tensor.
        full_shape : tuple
            Shape of the full tensor (``TensorTT.reconstruct.shape``). Makes the reconstruction process easier.
        """
        super(TensorTT, self).__init__()
        self._core_values = core_values.copy()
        self.full_shape = full_shape

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorTT
        """
        new_object = super(TensorTT, self).copy()
        return new_object

    def core(self, i):
        """ Specific core of the TensorTT representation

        Parameters
        ----------
        i : int
            Should not exceed the order of ``TensorTT.order - 1`` representation

        Returns
        -------
        core_tensor : Tensor
        """
        core_tensor = Tensor(self._core_values[i])
        return  core_tensor

    @property
    def cores(self):
        """ All cores of the TensorTT representation

        Returns
        -------
        core_list : list[Tensor]
        """
        core_list = [self.core(i) for i in range(len(self._core_values))]
        return core_list

    @property
    def order(self):
        """ Order of a tensor represented through the TT

        Returns
        -------
        order : int
        """
        return len(self._core_values)

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
        return tuple([core_values.shape[-1] for core_values in self._core_values[:-1]])

    @property
    def reconstruct(self):
        """ Converts the TT representation of a tensor into a full tensor

        Returns
        -------
        tensor : Tensor
        """
        rank = self.rank + (1,)
        core = self.cores[0]
        data = core.data
        for i, core in enumerate(self.cores[1:]):
            shape_2d = [rank[i], rank[i+1]*self.full_shape[i+1]]
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
    tensor_B : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    residual : Tensor
    """
    if isinstance(tensor_B, TensorCPD) or isinstance(tensor_B, TensorTKD) or isinstance(tensor_B, TensorTT):
        residual = Tensor(tensor_A.data - tensor_B.reconstruct.data)
    elif isinstance(tensor_B, Tensor):
        residual = Tensor(tensor_A.data - tensor_B.data)
    else:
        raise TypeError('Unknown data type of the approximation')
    return residual