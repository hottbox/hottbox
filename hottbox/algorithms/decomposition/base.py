import numpy as np
import scipy.linalg
import scipy.sparse.linalg


class Decomposition(object):
    """
    This is general interface for all classes that describe tensor decompositions and provides a brief summary of
    the general attributes and properties
    """

    def __init__(self):
        pass

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        str
        """
        return self.__class__.__name__

    def decompose(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (Decomposition) class')

    @property
    def converged(self):
        raise NotImplementedError('Not implemented in base (Decomposition) class')

    def _init_fmat(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (Decomposition) class')

    def plot(self):
        raise NotImplementedError('Not implemented in base (Decomposition) class')


def svd(matrix, rank=None):
    """ Computes SVD on matrix

    Parameters
    ----------
    matrix : np.ndarray
    rank : int

    Returns
    -------
    U : np.ndarray
    S : np.ndarray
    V : np.ndarray

    """
    if matrix.ndim != 2:
        raise ValueError('Input should be a two-dimensional array. matrix.ndim is {} != 2'.format(matrix.ndim))
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if rank is None or rank >= min_dim:
        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix)
        U, S, V = U[:, :rank], S[:rank], V[:rank, :]
        return U, S, V

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(np.dot(matrix, matrix.T), k=rank, which='LM')
            S = np.sqrt(S)
            V = np.dot(matrix.T, U * 1 / S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(np.dot(matrix.T, matrix), k=rank, which='LM')
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1 / S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T
