"""
Classes for different tensor representations
"""

import numpy as np
from functools import reduce
from .operations import unfold, fold, mode_n_product
from ._meta import Mode, State


class Tensor(object):
    """ This class describes multidimensional data.

    All its methods implement all common operation on a tensor alone

    Attributes
    ----------
    _data : np.ndarray
        N-dimensional array.
    _modes : list[Mode]
        Description of the tensor modes in form of a list where each element is object of ``Mode`` class.
        Meta information.
    _state : State
        List of references to meta information about modes of a tensor.
        Meta information.
    """

    def __init__(self, array, custom_state=None, mode_names=None) -> None:
        """ Create object of ``Tensor`` class

        Parameters
        ----------
        array : np.ndarray
            N-dimensional array
        custom_state : dict

        mode_names : list[str]
            Description of the tensor modes.
            If nothing is specified then all modes of the created ``Tensor``
            get generic names 'mode-0', 'mode-1' etc.
        """
        # self._validate_init_data(array=array, mode_names=mode_names, custom_state=custom_state, ft_shape=ft_shape)
        self._data = array.copy()
        self._state, self._modes = self._create_meta(array=array,
                                                     custom_state=custom_state,
                                                     mode_names=mode_names)

    def __eq__(self, other):
        """
        Returns
        -------
        bool

        Notes
        -----
        Tensors are equal when everything is the same.
        """
        equal = False
        if isinstance(self, other.__class__):
            if self.shape == other.shape:
                data_equal = np.allclose(self.data, other.data,  rtol=1e-05, atol=1e-08, equal_nan=True)
                state_equal = self.state == other.state
                modes_equal = all([self.modes[i] == other.modes[i] for i in range(self.order)])
                equal = data_equal and state_equal and modes_equal

        return equal

    def __add__(self, other):
        """ Summation of objects of ``Tensor`` class

        Returns
        -------
        tensor : Tensor

        Notes
        -----
            Two objects of ``Tensor`` class can be added together if:
                1) Both are in normal state (haven't been rotated, unfolded or folded)
                2) Both have the same shape
                3) Bath have the same indices : all([self.modes[i].index == other.modes[i].index])
            If names of the modes are different the summation will be performed, and
        """
        if not isinstance(self, other.__class__):
            raise TypeError("Don't know how to sum object of {} class "
                            "with an object of {} class!".format(self.__class__.__name__,
                                                                 other.__class__.__name__))
        if not all([self.in_normal_state, other.in_normal_state]):
            raise ValueError("Both tensors should be in normal state!")
        if self.shape != other.shape:
            raise ValueError("Both tensors should have the same shape!")
        if not all([self.modes[i].index == other.modes[i].index for i in range(self.order)]):
            raise ValueError("Both tensors should have the same indices!")
        array = self.data + other.data
        tensor = Tensor(array=array).copy_modes(self)
        if self.mode_names != other.mode_names:
            for i in range(tensor.order):
                tensor.reset_mode_name(mode=i)
        return tensor

    def __str__(self):
        """ Provides general information about this instance."""
        return "This tensor is of order {} and consists of {} elements.\n" \
               "Sizes and names of its modes are {} and {} respectively.".format(self.order, self.size,
                                                                                 self.shape, self.mode_names)

    def __repr__(self):
        return str(self)

    @staticmethod
    def _validate_init_data(array, mode_names, custom_state, ft_shape):
        """ Validate data for ``Tensor`` constructor

        Parameters
        ----------
        array : np.ndarray
        mode_names : list[str]
        ft_shape : tuple
        """
        # validate data array
        if not isinstance(array, np.ndarray):
            raise TypeError('Input data should be a numpy array')

        # validate mode_names if provided
        if mode_names is not None:
            if not isinstance(mode_names, list):
                raise TypeError("You should use list for mode_names!")
            if array.ndim != len(mode_names):
                raise ValueError("Incorrect number of names for the modes of a tensor: {0} != {1} "
                                 "('array.ndim != len(mode_names)')!\n".format(array.ndim,
                                                                               len(mode_names)
                                                                               )
                                 )
            if not all(isinstance(name, str) for name in mode_names):
                raise TypeError("The list of mode names should contain only strings!")

        # validate ft_shape if provided
        if ft_shape is not None:
            if not isinstance(ft_shape, tuple):
                raise TypeError("Incorrect type of the parameter `ft_shape`!\n"
                                "It should be tuple")
            size = reduce(lambda x, y: x * y, ft_shape)
            if array.size != size:
                raise ValueError("Values of `ft_shape` are inconsistent with the provided data array ({} != {})!\n"
                                 "reduce(lambda x, y: x * y, ft_shape) != array.size".format(size, array.size))

    @staticmethod
    def _create_meta(array, custom_state, mode_names):
        """ Create meta data for the tensor

        Parameters
        ----------
        array : np.ndarray
        custom_state : dict
        mode_names : list[str]

        Returns
        -------
        state : State
            Meta information related to reshaping of the tensor
        modes : list[Mode]
            Meta information related to modes of the tensor
        """

        if custom_state is None:
            custom_state = dict(normal_shape=tuple([mode_size for mode_size in array.shape]),
                                mode_order=[[i] for i in range(array.ndim)])
        state = State(**custom_state)

        if mode_names is None:
            mode_names = ["mode-{}".format(i) for i in range(len(state._normal_shape))]
        modes = [Mode(name=name) for name in mode_names]
        return state, modes

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
        shape = self.state.normal_shape
        return shape

    @property
    def state(self):
        return self._state

    @property
    def modes(self):
        """ Meta data for the modes of a tensor

        Returns
        -------
        list[Mode]
        """
        return self._modes

    @property
    def in_normal_state(self):
        """ Checks state of a tensor

        Returns
        -------
        bool
            If True, then can call `unfold` and `mode_n_product`
            If False, then can call `fold`
        """
        return self.state.is_normal()

    @property
    def mode_names(self):
        """ Description of the tensor modes in current state

        Returns
        -------
        names : list[str]
        """
        if self.in_normal_state:
            # if tensor in the original
            names = [mode.name for mode in self.modes]
        else:
            # if tensor is in unfolded state
            state_0 = self.state.mode_order[0]
            state_1 = self.state.mode_order[1]
            name_0 = self.modes[state_0[0]].name
            name_2 = [self.modes[i].name for i in state_1]
            name_1 = '_'.join(name_2)
            names = [name_0, name_1]
        return names

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
        """ Shape of a tensor in current state

        Returns
        -------
        tuple
            Sizes of all dimensions of a tensor
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

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : Tensor
            New object of Tensor class with attributes having the same values, but no memory space is shared
        """
        array = self.data
        custom_state = dict(normal_shape=self.state._normal_shape,
                            mode_order=self.state._mode_order)
        new_object = Tensor(array=array, custom_state=custom_state)
        # In order to preserved index if it was specified
        new_object.copy_modes(self)
        return new_object

    def copy_modes(self, tensor):
        """ Copy modes meta from tensor representation

        Parameters
        ----------
        tensor : {Tensor, TensorCPD, TensorTKD, TensorTT}

        Returns
        -------
        self : Tensor
        """
        self._modes = [mode.copy() for mode in tensor.modes]
        return self

    def reset_meta(self):
        """ Set all meta information with respect to the current shape of data array

        Returns
        -------
        self
        """
        default_mode_names = ["mode-{}".format(i) for i in range(self.data.ndim)]
        self._modes = [Mode(name=name) for name in default_mode_names]
        self._state.reset()
        return self

    def set_mode_names(self, mode_names):
        """ Rename modes of a tensor
        
        Parameters
        ----------        
        mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Returns
        -------
        self : Tensor
            Return self so that methods could be chained
        """
        if len(mode_names.keys()) > self.order:
            raise ValueError("Too many mode names have been specified")
        if not all(isinstance(mode, int) for mode in mode_names.keys()):
            raise TypeError("The dict of `mode_names` should contain only integer keys!")
        if not all(mode < self.order for mode in mode_names.keys()):
            raise ValueError("All specified mode values should not exceed the order of the tensor!")
        if not all(mode >= 0 for mode in mode_names.keys()):
            raise ValueError("All specified mode keys should be non-negative!")

        for i, name in mode_names.items():
            self.modes[i].set_name(name=name)

        return self

    def reset_mode_name(self, mode=None):
        """ Set default name for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which name to be set to default value
            By default resets names of all modes

        Returns
        -------
        self
        """
        if mode is None:
            for i, t_mode in enumerate(self.modes):
                default_name = "mode-{}".format(i)
                t_mode.set_name(name=default_name)
        else:
            default_name = "mode-{}".format(mode)
            self.modes[mode].set_name(name=default_name)
        return self

    def set_mode_index(self, mode_index):
        """ Set index for specified mode

        Parameters
        ----------
        mode_index : dict
            New indices for the tensor modes in form of a dictionary.
            Key defines the mode whose index to be changed.
            Value contains a list of new indices for this mode.

        Returns
        -------
        self
        """
        if len(mode_index.keys()) > self.order:
            raise ValueError("Too many sets of indices have been specified")
        if not all(isinstance(mode, int) for mode in mode_index.keys()):
            raise TypeError("The dict of `mode_index` should contain only integer keys!")
        if not all(mode < self.order for mode in mode_index.keys()):
            raise ValueError("All specified mode values should not exceed the order of the tensor!")
        if not all(mode >= 0 for mode in mode_index.keys()):
            raise ValueError("All specified mode keys should be non-negative!")
        if not all([len(index) == self.ft_shape[mode] for mode, index in mode_index.items()]):
            raise ValueError("Not enough of too many indices for the specified mode")

        for i, index in mode_index.items():
            self.modes[i].set_index(index=index)

        return self

    def reset_mode_index(self, mode=None):
        """ Drop index for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which index to be dropped
            By default resets all indices

        Returns
        -------
        self
        """
        if mode is None:
            for i in range(self.order):
                self.modes[i].reset_index()
        else:
            self.modes[mode].reset_index()
        return self

    def describe(self):
        """ Provides some statistics of data for this instance."""
        pass

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
            Unfolding operation does not change `_modes` attribute but changes `_state._mode_order` attribute
        """
        if not self.in_normal_state:
            raise TypeError("The tensor is not in the original form")

        # Unfold data
        data_unfolded = unfold(self.data, mode)

        if inplace:
            tensor = self
        else:
            tensor = self.copy()

        tensor._data = data_unfolded
        tensor.state.unfold(mode=mode)
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
            Folding operation does not change `_modes` attribute but changes `_state._mode_order` attribute
        """
        # Do not do anything if the tensor is in the normal form (hadn't been unfolded before)
        if self.in_normal_state:
            raise TypeError("The tensor hadn't bee unfolded before")

        # Fold data
        temp = self.state._mode_order[0]
        folding_mode = temp[0]
        data_folded = fold(matrix=self.data, mode=folding_mode, shape=self.ft_shape)

        if inplace:
            tensor = self
        else:
            tensor = self.copy()

        tensor._data = data_folded
        tensor.state.fold()
        return tensor

    def mode_n_product(self, matrix, mode, new_name=None, inplace=True):
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
            1. Mode-n product operation changes the `_state._normal_shape` attribute
            2. Remember that mode-n product changes the shape of the tensor. Presumably, it also changes
               the interpretation of that mode depending on the matrix
            3. If `matrix` is an object of `Tensor` class then you shouldn't specify `new_name`, since
               it will be changed to `matrix.mode_names[0]`
            4. If `matrix.mode_names[0] == "mode-0"` then no changes to `tensor.mode_names` will be made
        """
        if not self.in_normal_state:
            raise TypeError("The tensor is not in the original form")

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

        new_data = mode_n_product(tensor=self.data, matrix=matrix.data, mode=mode)
        new_normal_shape = new_data.shape
        if inplace:
            tensor = self
        else:
            tensor = self.copy()
        tensor._data = new_data
        tensor.state.set_normal_shape(new_normal_shape=new_normal_shape)
        tensor.reset_mode_index(mode=mode)

        # The only one case when mode name won't be changed
        if new_name != "mode-0":            
            new_mode_names = {mode: new_name}
            tensor.set_mode_names(mode_names=new_mode_names)

        return tensor


class BaseTensorTD(object):
    """
    This class provides a general interface for a tensor represented through a tensor decomposition.
    """
    def __init__(self):
        pass

    @staticmethod
    def _validate_init_data(**kwargs):
        """ Validate data for the constructor of a new object """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def _create_modes(self, mode_names):
        """ Create meta data for each mode of tensor representation

        Parameters
        ----------
        mode_names : list[str]

        Returns
        -------
        modes : list[Mode]
        """
        if mode_names is None:
            mode_names = ["mode-{}".format(i) for i in range(self.order)]
        modes = [Mode(name=name) for name in mode_names]
        return modes

    def copy(self):
        """ Produces a copy of itself as a new object """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def modes(self):
        """ Meta data for each mode of tensor representation """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def order(self):
        """ Order of a tensor in full form """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def rank(self):
        """ Rank of an efficient representation """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def size(self):
        """ Number of elements for efficient representation """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    @property
    def frob_norm(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def reconstruct(self):
        """ Convert to the full tensor as an object of Tensor class """
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def unfold(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def fold(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def mode_n_product(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTD) class')

    def copy_modes(self, tensor):
        """ Copy modes meta from tensor

        Parameters
        ----------
        tensor : {Tensor, TensorCPD, TensorTKD, TensorTT}

        Returns
        -------
        self
        """
        self._modes = [mode.copy() for mode in tensor.modes]
        return self

    def set_mode_names(self, mode_names):
        """ Rename modes of a tensor representation

        Parameters
        ----------
        mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Returns
        -------
        self
        """
        if len(mode_names.keys()) > self.order:
            raise ValueError("Too many mode names have been specified")
        if not all(isinstance(mode, int) for mode in mode_names.keys()):
            raise TypeError("The dict of `mode_names` should contain only integer keys!")
        if not all(mode < self.order for mode in mode_names.keys()):
            raise ValueError("All specified mode values should not exceed the order of the tensor!")
        if not all(mode >= 0 for mode in mode_names.keys()):
            raise ValueError("All specified mode keys should be non-negative!")

        for i, name in mode_names.items():
            self.modes[i].set_name(name)

        return self

    def reset_mode_name(self, mode):
        """ Set default name for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which name to be set to default value
            By default resets names of all modes

        Returns
        -------
        self
        """
        if mode is None:
            for i, t_mode in enumerate(self.modes):
                default_name = "mode-{}".format(i)
                t_mode.set_name(name=default_name)
        else:
            default_name = "mode-{}".format(mode)
            self.modes[mode].set_name(name=default_name)
        return self

    def set_mode_index(self, mode_index):
        """ Set index for specified mode

        Parameters
        ----------
        mode_index : dict
            New indices for the factor matrices in form of a dictionary.
            Key defines the mode whose index to be changed.
            Value contains a list of new indices for this mode.

        Returns
        -------
        self
        """
        if len(mode_index.keys()) > self.order:
            raise ValueError("Too many sets of indices have been specified")
        if not all(isinstance(mode, int) for mode in mode_index.keys()):
            raise TypeError("The dict of `mode_index` should contain only integer keys!")
        if not all(mode < self.order for mode in mode_index.keys()):
            raise ValueError("All specified mode values should not exceed the order of the tensor!")
        if not all(mode >= 0 for mode in mode_index.keys()):
            raise ValueError("All specified mode keys should be non-negative!")
        if isinstance(self, TensorTT):
            index_long_enough = all([len(index) == self._ft_shape[mode] for mode, index in mode_index.items()])
        else:
            index_long_enough = all([len(index) == self.fmat[mode].shape[0] for mode, index in mode_index.items()])
        if not index_long_enough:
            raise ValueError("Not enough of too many indices for the specified mode")

        for i, index in mode_index.items():
            self.modes[i].set_index(index=index)

        return self

    def reset_mode_index(self, mode):
        """ Drop index for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which index to be dropped
            By default resets all indices

        Returns
        -------
        self
        """
        if mode is None:
            for i in range(self.order):
                self.modes[i].reset_index()
        else:
            self.modes[mode].reset_index()
        return self


class TensorCPD(BaseTensorTD):
    """ Representation of a tensor in the CPD form.

    Attributes
    ----------
    _fmat : list[np.ndarray]
        Placeholder for a list of factor matrices for the CP representation of a tensor
    _core_values : np.ndarray
        Placeholder for an array of coefficients on the super-diagonal of a core for the CP representation of a tensor
    _modes : list[Mode]
        Description of the factor matrix for the corresponding mode
    """
    def __init__(self, fmat, core_values, mode_names=None):
        """ Create object of ``TensorCPD`` class
        
        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the CP representation of a tensor
        core_values : np.ndarray
            Array of coefficients on the super-diagonal of a core for the CP representation of a tensor
        mode_names : list[str]
            List of names for the factor matrices
        """
        super(TensorCPD, self).__init__()
        self._validate_init_data(fmat=fmat, core_values=core_values)
        self._fmat = [mat.copy() for mat in fmat]
        self._core_values = core_values.copy()
        self._modes = self._create_modes(mode_names=mode_names)

    @staticmethod
    def _validate_init_data(fmat, core_values):
        """ Validate data for the TensorCPD constructor

        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the CP representation of a tensor
        core_values : np.ndarray
            Array of coefficients on the super-diagonal of a core for the CP representation of a tensor
        """
        if not isinstance(core_values, np.ndarray):
            raise TypeError("Core values (`core_values`) should be a numpy array")
        if not isinstance(fmat, list):
            raise TypeError("All factor matrices (`fmat`) should be passed as a list!")
        for mat in fmat:
            if not isinstance(mat, np.ndarray):
                raise TypeError("Each of the factor matrices should be a numpy array!")
            if mat.ndim != 2:
                raise ValueError("Each of the factor matrices should a 2-dimensional numpy array!")

        kryskal_rank = len(core_values)
        if not all([mat.shape[1] == kryskal_rank for mat in fmat]):
            raise ValueError("Dimension mismatch!\n"
                             "Number of columns of all factor matrices should be the same and equal to len(core_values)!")

    def _create_modes(self, mode_names):
        """ Create meta data for each factor matrix

        Parameters
        ----------
        mode_names : list[str]

        Returns
        -------
        modes : list[Mode]
        """
        modes = super(TensorCPD, self)._create_modes(mode_names=mode_names)
        return modes

    @property
    def core(self):
        """ Core tensor of the CP representation of a tensor

        Returns
        -------
        core_tensor : Tensor
        """
        core_shape = self.rank * self.order
        core_tensor = super_diag_tensor(core_shape, values=self._core_values)
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
    def modes(self):
        """ Meta data for the factor matrices

        Returns
        -------
        list[Mode]
        """
        return self._modes

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

    def reconstruct(self, keep_meta=0):
        """ Converts the CP representation of a tensor into a full tensor

        Parameters
        ----------
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor : Tensor
        """
        tensor = self.core
        for mode, fmat in enumerate(self.fmat):
            tensor.mode_n_product(fmat, mode=mode, inplace=True)

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(self.modes)}
            tensor.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor.copy_modes(self)
        else:
            pass

        return tensor

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorCPD
        """
        fmat = self._fmat
        core_values = self._core_values
        new_object = TensorCPD(fmat=fmat, core_values=core_values)
        new_object.copy_modes(self)
        return new_object

    def copy_modes(self, tensor):
        """ Copy modes meta from tensor

        Parameters
        ----------
        tensor : {Tensor, TensorCPD, TensorTKD, TensorTT}

        Returns
        -------
        self : TensorCPD

        Notes
        -----
            Most of the time this method should only be used by the CPD type algorithm
        """
        # TODO: check for dimensionality
        super(TensorCPD, self).copy_modes(tensor=tensor)
        return self

    def set_mode_names(self, mode_names):
        """ Rename modes of a tensor representation

        Parameters
        ----------
        mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Returns
        -------
        self : TensorCPD
        """
        super(TensorCPD, self).set_mode_names(mode_names=mode_names)

        return self

    def reset_mode_name(self, mode=None):
        """ Set default name for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which name to be set to default value
            By default resets names of all modes

        Returns
        -------
        self : TensorCPD
        """
        super(TensorCPD, self).reset_mode_name(mode=mode)
        return self

    def set_mode_index(self, mode_index):
        """ Set index for specified mode

        Parameters
        ----------
        mode_index : dict
            New indices for the factor matrices in form of a dictionary.
            Key defines the mode whose index to be changed.
            Value contains a list of new indices for this mode.

        Returns
        -------
        self : TensorCPD
        """
        super(TensorCPD, self).set_mode_index(mode_index=mode_index)
        return self

    def reset_mode_index(self, mode=None):
        """ Drop index for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which index to be dropped
            By default resets all indices

        Returns
        -------
        self : TensorCPD
        """
        super(TensorCPD, self).reset_mode_index(mode=mode)
        return self


class TensorTKD(BaseTensorTD):
    """ Representation of a tensor in the Tucker form.

    Attributes
    ----------
    _fmat : list[np.ndarray]
        Placeholder for a list of factor matrices for the Tucker representation of a tensor
    _core_values : np.ndarray
        Placeholder for a core of the Tucker representation of a tensor
    _modes : list[Mode]
        Description of the factor matrix for the corresponding mode
    """
    def __init__(self, fmat, core_values, mode_names=None):
        """ Create object of ``TensorTKD`` class
        
        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the Tucker representation of a tensor
        core_values : np.ndarray
            Core of the Tucker representation of a tensor
        mode_names : list[str]
            List of names for the factor matrices
        """
        super(TensorTKD, self).__init__()
        self._validate_init_data(fmat=fmat, core_values=core_values)
        self._fmat = [mat.copy() for mat in fmat]
        self._core_values = core_values.copy()
        self._modes = self._create_modes(mode_names=mode_names)

    @staticmethod
    def _validate_init_data(fmat, core_values):
        """ Validate data for the TensorTKD constructor

        Parameters
        ----------
        fmat : list[np.ndarray]
            List of factor matrices for the Tucker representation of a tensor
        core_values : np.ndarray
            Core of the Tucker representation of a tensor
        """
        if not isinstance(core_values, np.ndarray):
            raise TypeError("Core values (`core_values`) should be a numpy array")
        if not isinstance(fmat, list):
            raise TypeError("All factor matrices (`fmat`) should be passed as a list!")
        for mat in fmat:
            if not isinstance(mat, np.ndarray):
                raise TypeError("Each of the factor matrices should be a numpy array!")
            if mat.ndim != 2:
                raise ValueError("Each of the factor matrices should a 2-dimensional numpy array!")

        ml_rank = core_values.shape
        order = core_values.ndim
        if len(fmat) != order:
            raise ValueError("Not enough or too many factor matrices for the specified core tensor!\n"
                             "{}!={} (`len(fmat) != core_values.ndim`)".format(len(fmat), order))
        mat_shapes = tuple([mat.shape[1] for mat in fmat])
        if not all([mat_shapes[i] == ml_rank[i] for i in range(order)]):
            raise ValueError("Dimension mismatch between the factor matrices and the core tensor!\n"
                             "The number of columns of a factor matrix should match the corresponding "
                             "dimension size of the core tensor:\n"
                             "{} != {} (fmat[i].shape[1] != core_values.shape)".format(mat_shapes, ml_rank))

    def _create_modes(self, mode_names):
        """ Create meta data for each factor matrix

        Parameters
        ----------
        mode_names : list[str]

        Returns
        -------
        modes : list[Mode]
        """
        modes = super(TensorTKD, self)._create_modes(mode_names=mode_names)
        return modes

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
    def modes(self):
        """ Meta data for the factor matrices

        Returns
        -------
        list[Mode]
        """
        return self._modes

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

    def reconstruct(self, keep_meta=0):
        """ Converts the Tucker representation of a tensor into a full tensor

        Parameters
        ----------
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor : Tensor
        """
        tensor = self.core
        for mode, fmat in enumerate(self.fmat):
            tensor.mode_n_product(fmat, mode=mode, inplace=True)

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(self.modes)}
            tensor.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor.copy_modes(self)
        else:
            pass

        return tensor

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorTKD
        """
        fmat = self._fmat
        core_values = self._core_values
        new_object = TensorTKD(fmat=fmat, core_values=core_values)
        new_object.copy_modes(self)
        return new_object

    def copy_modes(self, tensor):
        """ Copy modes meta from tensor

        Parameters
        ----------
        tensor : {Tensor, TensorCPD, TensorTKD, TensorTT}

        Returns
        -------
        self : TensorTKD

        Notes
        -----
            Most of the time this method should only be used by the CPD type algorithm
        """
        # TODO: check for dimensionality
        super(TensorTKD, self).copy_modes(tensor=tensor)
        return self

    def set_mode_names(self, mode_names):
        """ Rename modes of a tensor representation

        Parameters
        ----------
        mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Returns
        -------
        self : TensorTKD
        """
        super(TensorTKD, self).set_mode_names(mode_names=mode_names)

        return self

    def reset_mode_name(self, mode=None):
        """ Set default name for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which name to be set to default value
            By default resets names of all modes

        Returns
        -------
        self : TensorTKD
        """
        super(TensorTKD, self).reset_mode_name(mode=mode)
        return self

    def set_mode_index(self, mode_index):
        """ Set index for specified mode

        Parameters
        ----------
        mode_index : dict
            New indices for the factor matrices in form of a dictionary.
            Key defines the mode whose index to be changed.
            Value contains a list of new indices for this mode.

        Returns
        -------
        self : TensorTKD
        """
        super(TensorTKD, self).set_mode_index(mode_index=mode_index)
        return self

    def reset_mode_index(self, mode=None):
        """ Drop index for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which index to be dropped
            By default resets all indices

        Returns
        -------
        self : TensorTKD
        """
        super(TensorTKD, self).reset_mode_index(mode=mode)
        return self


class TensorTT(BaseTensorTD):
    """ Representation of a tensor in the TT form.

    Attributes
    ----------
    _core_values : list[np.ndarray]
        Placeholder for a list of cores for the Tensor Train representation of a tensor.
    _ft_shape : tuple
        Placeholder for a shape of the full tensor (``TensorTT.reconstruct.shape``).
        Makes the reconstruction process easier.
    """
    def __init__(self, core_values, ft_shape, mode_names=None):
        """
        
        Parameters
        ----------
        core_values : list[np.ndarray]
            List of cores for the Tensor Train representation of a tensor.
        ft_shape : tuple
            Shape of the full tensor (``TensorTT.reconstruct.shape``). Makes the reconstruction process easier.
        """
        super(TensorTT, self).__init__()
        self._validate_init_data(core_values=core_values, ft_shape=ft_shape)
        self._core_values = [core.copy() for core in core_values]
        self._ft_shape = tuple([mode_size for mode_size in ft_shape])
        self._modes = self._create_modes(mode_names=mode_names)

    def _validate_init_data(self, core_values, ft_shape):
        """ Validate data for the TensorTT constructor

        Parameters
        ----------
        core_values : list[np.ndarray]
            List of cores for the Tensor Train representation of a tensor.
        ft_shape : tuple
            Shape of the full tensor (``TensorTT.reconstruct.shape``). Makes the reconstruction process easier.
        """
        # validate types of the input data
        if not isinstance(ft_shape, tuple):
            raise TypeError("The parameter `ft_shape` should be passed as tuple!")
        if not isinstance(core_values, list):
            raise TypeError("The parameter `core_values` should be passed as list!")
        for core in core_values:
            if not isinstance(core, np.ndarray):
                raise TypeError("Each element from `core_values` should be a numpy array!")

        # validate the shape of the tensor in full format
        # TODO: remove this validation when `self._ft_shape` will be removed
        if len(core_values) != len(ft_shape):
            # raise ValueError("Inconsistent shape of the tensor in full form!")
            raise ValueError("Not enough or too many cores for the specified shape of the tensor in full form:\n"
                             "{} != {} (len(core_values) != len(ft_shape))".format(len(core_values), len(ft_shape)))

        # validate sizes of the cores
        if ((core_values[0].ndim != 2) or (core_values[-1].ndim != 2)):
            raise ValueError("The first and the last elements of the `core_values` "
                             "should be 2-dimensional numpy arrays!")
        for i in range(1, len(core_values) - 1):
            if core_values[i].ndim != 3:
                raise ValueError("All but first and the last elements of the `core_values` "
                                 "should be 3-dimensional numpy arrays!")
        for i in range(len(core_values)-1):
            if core_values[i].shape[-1] != core_values[i+1].shape[0]:
                raise ValueError("Dimension mismatch for the specified cores:\n"
                                 "Last dimension of core_values[{}] should be the same as the "
                                 "first dimension of core_values[{}]".format(i, i+1))

        # validate the shape of the tensor in full format
        # TODO: remove this validation when `self._ft_shape` will be removed
        extracted_full_shape = [None] * len(core_values)
        extracted_full_shape[0] = core_values[0].shape[0]
        extracted_full_shape[-1] = core_values[-1].shape[1]
        for i in range(1, len(core_values)-1):
            extracted_full_shape[i] = core_values[i].shape[1]

        if tuple(extracted_full_shape) != ft_shape:
            raise ValueError("Inconsistent shape of the tensor in full form:\n"
                             "{} != {} (extracted_full_shape != ft_shape)".format(extracted_full_shape, ft_shape))

    def _create_modes(self, mode_names):
        """ Create meta data for each factor matrix

        Parameters
        ----------
        mode_names : list[str]

        Returns
        -------
        modes : list[Mode]
        """
        modes = super(TensorTT, self)._create_modes(mode_names=mode_names)
        return modes

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : TensorTT
        """
        core_values = self._core_values
        ft_shape = self._ft_shape
        new_object = TensorTT(core_values=core_values, ft_shape=ft_shape)
        new_object.copy_modes(self)
        return new_object

    def core(self, i):
        """ Specific core of the TensorTT representation

        Parameters
        ----------
        i : int
            Should not exceed the order of ``TensorTT.order`` representation

        Returns
        -------
        core_tensor : Tensor
        """
        if abs(i) >= self.order:
            raise IndexError("List index out of range!\n"
                             "Index for the core of interest cannot exceed the order of TT representation: "
                             "abs({}) >= {} (abs(i) >= self.order)".format(i, self.order))
        core_tensor = Tensor(self._core_values[i])
        return core_tensor

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
    def modes(self):
        """ Meta data for the factor matrices

        Returns
        -------
        list[Mode]
        """
        return self._modes

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

    def reconstruct(self, keep_meta=0):
        """ Converts the TT representation of a tensor into a full tensor

        Parameters
        ----------
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor : Tensor
        """
        rank = self.rank + (1,)
        core = self.cores[0]
        data = core.data
        for i, core in enumerate(self.cores[1:]):
            shape_2d = [rank[i], rank[i+1] * self._ft_shape[i + 1]]
            core_flat = np.reshape(core.data, shape_2d, order='F')
            data = np.reshape(data, [-1, rank[i]], order='F')
            data = np.dot(data, core_flat)
        data = np.reshape(data, self._ft_shape, order='F')
        tensor = Tensor(data)

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(self.modes)}
            tensor.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor.copy_modes(self)
        else:
            pass

        return tensor

    def copy_modes(self, tensor):
        """ Copy modes meta from tensor

        Parameters
        ----------
        tensor : {Tensor, TensorCPD, TensorTKD, TensorTT}

        Returns
        -------
        self : TensorTT

        Notes
        -----
            Most of the time this method should only be used by the CPD type algorithm
        """
        # TODO: check for dimensionality
        super(TensorTT, self).copy_modes(tensor=tensor)
        return self

    def set_mode_names(self, mode_names):
        """ Rename modes of a tensor representation

        Parameters
        ----------
        mode_names : dict
            New names for the tensor modes in form of a dictionary
            The name of the mode defined by the Key of the dict will be renamed to the corresponding Value

        Returns
        -------
        self : TensorTT
        """
        super(TensorTT, self).set_mode_names(mode_names=mode_names)

        return self

    def reset_mode_name(self, mode=None):
        """ Set default name for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which name to be set to default value
            By default resets names of all modes

        Returns
        -------
        self : TensorTT
        """
        super(TensorTT, self).reset_mode_name(mode=mode)
        return self

    def set_mode_index(self, mode_index):
        """ Set index for specified mode

        Parameters
        ----------
        mode_index : dict
            New indices for the factor matrices in form of a dictionary.
            Key defines the mode whose index to be changed.
            Value contains a list of new indices for this mode.

        Returns
        -------
        self : TensorTT
        """
        super(TensorTT, self).set_mode_index(mode_index=mode_index)
        return self

    def reset_mode_index(self, mode=None):
        """ Drop index for the specified mode number

        Parameters
        ----------
        mode : int
            Mode number which index to be dropped
            By default resets all indices

        Returns
        -------
        self : TensorTT
        """
        super(TensorTT, self).reset_mode_index(mode=mode)
        return self


def super_diag_tensor(shape, values=None):
    """ Super-diagonal tensor of the specified `order`.

    Parameters
    ----------
    shape : tuple
        Desired shape of the tensor.
        len(shape) defines the order of the tensor, whereas its values specify sizes of dimensions of the tensor.
    values : np.ndarray
        Array of values on the super-diagonal of a tensor. By default contains only ones.
        Length of this vector defines Kryskal rank which is equal to `shape[0]`.

    Returns
    -------
    tensor : Tensor
    """
    if not isinstance(shape, tuple):
        raise TypeError("Parameter `shape` should be passed as a tuple!")
    if not all(mode_size == shape[0] for mode_size in shape):
        raise ValueError("All values in `shape` should have the same value!")

    order = len(shape)
    rank = shape[0]
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

    if isinstance(tensor_approx, Tensor):
        residual = Tensor(tensor_orig.data - tensor_approx.data)
    elif isinstance(tensor_approx, TensorCPD):
        residual = Tensor(tensor_orig.data - tensor_approx.reconstruct().data)
    elif isinstance(tensor_approx, TensorTKD):
        residual = Tensor(tensor_orig.data - tensor_approx.reconstruct().data)
    elif isinstance(tensor_approx, TensorTT):
        residual = Tensor(tensor_orig.data - tensor_approx.reconstruct().data)
    else:
        raise TypeError("Unknown data type of the approximation tensor!\n"
                        "The available types for `tensor_B` are `Tensor`,  `TensorCPD`,  `TensorTKD`,  `TensorTT`")
    return residual
