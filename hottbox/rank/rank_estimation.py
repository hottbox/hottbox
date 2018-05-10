import numpy as np
from ..core.structures import Tensor
from ..algorithms.decomposition.cpd import CPD


def rankest(tensor, rank_range, epsilon=10e-3, verbose=False):
    """ Estimate the optimal Kryskal rank of a tensor

    Parameters
    ----------
    tensor : Tensor
        Multidimensional data which Kryskal rank is to be estimated
    rank_range : list[int]
        List of rank values to be tested
    epsilon : float
        Threshold for the relative error of approximation.
    verbose : bool
        Enable verbose output

    Returns
    -------
    optimal_rank : tuple
        Optimal kryskal rank. For consistency, the type of the returned value is tuple
    """
    if not isinstance(rank_range, list):
        raise TypeError("The `rank_range` should be passed as a list of integers")
    if not all(isinstance(value, int) for value in rank_range):
        raise TypeError("The `rank_range` should consist of integers only")

    cpd = CPD(verbose=False)
    rel_error = []
    for rank in rank_range:
        cpd.decompose(tensor=tensor, rank=(rank,))
        rel_error.append(cpd.cost[-1])
        if verbose:
            print('Rank = {}, Approximation error = {}'.format((rank,), cpd.cost[-1]))
        if rel_error[-1] <= epsilon:
            break
        # Reset cost value for cpd. Should work even without it
        cpd.cost = []
    optimal_value = rank_range[rel_error.index(min(rel_error))]
    optimal_rank = (optimal_value,)
    return optimal_rank


def mlrank(tensor):
    """ Calculate the multilinear rank of a tensor

    Parameters
    ----------
    tensor : Tensor
        Multidimensional data which multilinear rank is to be computed

    Returns
    -------
    ml_rank : tuple
        Multilinear rank
    """
    # TODO: implement setting a threshold for singular values
    order = tensor.order
    ml_rank = [np.linalg.matrix_rank(tensor.unfold(mode=i, inplace=False).data) for i in range(order)]
    ml_rank = tuple(ml_rank)
    return ml_rank
