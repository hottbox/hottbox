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

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.__dict__,
                                               offset=len(class_name)+4, ),)

    def copy(self):
        """ Copy of the Decomposition as a new object """
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

    def decompose(self, **kwargs):
        raise NotImplementedError('Not implemented in base (Decomposition) class')

    @property
    def converged(self):
        raise NotImplementedError('Not implemented in base (Decomposition) class')

    def _init_fmat(self, **kwargs):
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


def _pprint(params, offset=0, printer=repr):
    """ Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int
        The offset in characters to add at the begin of each line.

    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr

    Notes
    -----
    Implementation is taken from ``sklearn.base._pprint`` with minor modifications
    to avoid additional dependencies.
    """
    # Do a multi-line justified repr:
    param_names = [p for p in params.keys() if p is not "cost"]
    param_names.sort()

    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, name in enumerate(param_names):
        value = params[name]
        if isinstance(value, float):
            this_repr = '%s=%s' % (name, str(value))
        else:
            this_repr = '%s=%s' % (name, printer(value))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)
    # options = np.get_printoptions()
    # np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
