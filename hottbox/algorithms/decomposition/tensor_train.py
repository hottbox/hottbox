import warnings
import scipy
import numpy as np
from .base import Decomposition
from ...core.structures import Tensor, TensorTT, residual_tensor


class BaseTensorTrain(Decomposition):

    def __init__(self, verbose):
        super(BaseTensorTrain, self).__init__()
        self.verbose = verbose

    def copy(self):
        """ Copy of the Decomposition as a new object """
        new_object = super(BaseTensorTrain, self).copy()
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(BaseTensorTrain, self).name
        return decomposition_name

    @property
    def converged(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTrain) class')

    def decompose(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (BaseTensorTrain) class')

    def _init_fmat(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (BaseTensorTrain) class')

    def plot(self):
        raise NotImplementedError('Not implemented in base (BaseTensorTrain) class')


class TTSVD(BaseTensorTrain):
    """ Tensor Train Decomposition.

    Parameters
    ----------

    verbose : bool
    """

    def __init__(self, verbose=False) -> None:
        super(TTSVD, self).__init__(verbose=verbose)

    def copy(self):
        """ Copy of the Decomposition as a new object """
        new_object = super(TTSVD, self).copy()
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(TTSVD, self).name
        return decomposition_name

    def decompose(self, tensor, rank, keep_meta=0):
        """ Performs TT-SVD on the `tensor` with respect to the specified `rank`

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired tt-rank for the given `tensor`
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor_tt : TensorTT
            Tensor train representation of the `tensor`

        Notes
        -----
        Reshaping of the data is performed with respect to the FORTRAN ordering. This makes it easy to compare results
        with the MATLAB implementation by Oseledets. This doesn't really matter (apart from time it takes to compute),
        as long as we do exactly the opposite for the reconstruction
        """
        # TODO: implement using C ordering for the reshape
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be an object of `Tensor` class!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")

        # since we consider core tensors to be only of order 3
        if (tensor.order - 1) != len(rank):
            raise ValueError("Incorrect number of values in `rank`:\n"
                             "{} != {} (tensor.order-1 != len(rank))".format(tensor.order, len(rank)))
        # since TT decomposition should compress data
        if any(rank[i] > tensor.shape[i] for i in range(len(rank))):
            raise ValueError("Some values in `rank` are greater then the corresponding mode sizes of a `tensor`:\n"
                             "{} > {} (rank > tensor.shape)".format(rank, tensor.shape[:-1]))
        if rank[-1] > tensor.shape[-1]:
            raise ValueError("The last value in `rank` is greater then the last mode size of a `tensor`:\n"
                             "{} > {} (rank[-1] > tensor.shape[-1])".format(rank[-1], tensor.shape[-1]))

        cores = []
        sizes = tensor.shape
        rank = (1,) + rank + (1,)
        C = tensor.data
        for k in range(tensor.order-1):
            rows = rank[k] * sizes[k]
            C = np.reshape(C, [rows, -1], order='F')
            U, S, V = _svd_tt(C, rank[k + 1])
            # Shouldn't slow down much since order of tensors is not big in general
            if k == 0:
                new_core = np.reshape(U, [sizes[k], rank[k+1]], order='F')
            else:
                new_core = np.reshape(U, [rank[k], sizes[k], rank[k+1]], order='F')
            cores.append(new_core)
            C = np.dot(V, np.diag(S)).T
        new_core = C
        cores.append(new_core)
        tensor_tt = TensorTT(core_values=cores)
        if self.verbose:
            residual = residual_tensor(tensor, tensor_tt)
            print('Relative error of approximation = {}'.format(abs(residual.frob_norm / tensor.frob_norm)))
        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(tensor.modes)}
            tensor_tt.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor_tt.copy_modes(tensor)
        else:
            pass
        return tensor_tt

    @property
    def converged(self):
        warnings.warn(
            "The {} algorithm is not iterative algorithm.\n"
            "Returning default value (True).".format(self.name), RuntimeWarning
        )
        return True

    def _init_fmat(self, tensor, rank):
        print("The {} algorithm does not required initialisation of factor matrices")

    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))


def _svd_tt(matrix, rank):
    U, S, V = scipy.linalg.svd(matrix)
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]  # V is transposed of what it should be
    return U, S, V.T
