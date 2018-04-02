import warnings
import scipy
import numpy as np
from .base import Decomposition
from ...core.structures import Tensor, TensorTT


class BaseTensorTrain(Decomposition):

    def __init__(self, verbose, mode_description):
        super(BaseTensorTrain, self).__init__()
        self.verbose = verbose
        self.mode_description = mode_description

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

    mode_description : str
    verbose : bool
    """

    def __init__(self, verbose=False, mode_description='mode_tt_svd') -> None:
        super(TTSVD, self).__init__(verbose=verbose,
                                    mode_description=mode_description)

    def decompose(self, tensor, rank):
        """ Performs TT-SVD on the `tensor` with respect to the specified `rank`

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired tt-rank for the given `tensor`

        Returns
        -------
        tensor_tt : TensorTT
            Tensor train representation of the `tensor`

        Notes
        -----
        Reshaping of the data is performed with respect to the FORTRAN ordering. This makes it easy to compare results
        with the MATLAB implementation by Oseledets. This doesn't really matter, as long as we do exactly the opposite
        for the reconstruction
        """
        # TODO: check that rank does not contain ones. check that the length of rank does not exceed order of a tensor
        # TODO: implement using C ordering for the reshape
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
            cores.append(Tensor(new_core))
            C = np.dot(V, np.diag(S)).T
        new_core = C
        cores.append(Tensor(new_core))
        tensor_tt = TensorTT(cores=cores, full_shape=tensor.shape)
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
