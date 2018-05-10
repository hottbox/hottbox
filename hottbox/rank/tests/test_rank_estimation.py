"""
Tests for the rank estimation module
"""
import pytest
import sys
import io
import numpy as np
from functools import reduce
from ..rank_estimation import *
from ...core.structures import Tensor


def test_rankest():
    """ Tests for rankest"""
    captured_output = io.StringIO()     # Create StringIO object for testing verbosity
    sys.stdout = captured_output        # and redirect stdout.

    # ------ tests when optimal rank is within the provided range of values
    shape = (3, 3, 3)
    size = reduce(lambda x, y: x * y, shape)
    np.random.seed(0)
    tensor = Tensor(np.random.rand(size).reshape(shape))
    rank_range = [i for i in range(2, 10)]
    true_rank = (5,)

    kryskal_rank = rankest(tensor, rank_range)
    assert isinstance(kryskal_rank, tuple)
    assert kryskal_rank == true_rank
    assert captured_output.getvalue() == ''  # to check that nothing was printed

    rankest(tensor, rank_range, verbose=True)
    assert captured_output.getvalue() != ''  # to check that something was actually printed

    # ------ tests when optimal rank is not within the provided range of values
    #        therefore all values have been tested (this also ensures full coverage of rankest)
    shape = (4, 4, 4)
    size = reduce(lambda x, y: x * y, shape)
    np.random.seed(0)
    tensor = Tensor(np.random.rand(size).reshape(shape))
    rank_range = [i for i in range(2, 9)]

    true_rank = (rank_range[-1],)
    kryskal_rank = rankest(tensor, rank_range)
    assert kryskal_rank == true_rank

    # test for changing default value for the threshold
    new_threshold = 0.1
    true_rank = (6,)
    kryskal_rank = rankest(tensor, rank_range, epsilon=new_threshold)
    assert kryskal_rank == true_rank

    # rank_range should be passed as a list
    with pytest.raises(TypeError):
        incorrect_rank_range = range(10)
        rankest(tensor, rank_range=incorrect_rank_range)

    # rank_range should contain only integers
    with pytest.raises(TypeError):
        incorrect_rank_range = [1, 2.5, 3]
        rankest(tensor, rank_range=incorrect_rank_range)


def test_mlrank():
    """ Tests for mlrank """
    shape = (5, 4, 5)
    size = reduce(lambda x, y: x * y, shape)
    np.random.seed(0)
    tensor = Tensor(np.random.rand(size).reshape(shape))
    tensor_mlrank = mlrank(tensor)
    assert isinstance(tensor_mlrank, tuple)
