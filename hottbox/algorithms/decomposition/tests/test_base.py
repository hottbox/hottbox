"""
Tests for the base module of decomposition algorithms
"""
import pytest
import sys
import io
import numpy as np
from functools import reduce
from ..base import *


class TestDecomposition:
    """ Tests for the Decomposition as an interface"""

    def test_init(self):
        decomposition_interface = Decomposition()
        with pytest.raises(NotImplementedError):
            decomposition_interface.decompose()
        with pytest.raises(NotImplementedError):
            decomposition_interface.converged()
        with pytest.raises(NotImplementedError):
            decomposition_interface.plot()
        with pytest.raises(NotImplementedError):
            decomposition_interface._init_fmat()


def test_svd():
    """ Tests for svd function """
    np.random.seed(0)
    with pytest.raises(ValueError):
        matrix = np.ones((2, 2, 2))
        svd(matrix)

    matrix = np.random.randn(4, 6)
    result = svd(matrix=matrix)
    assert result[0].shape == (matrix.shape[0], matrix.shape[0])
    assert result[1].shape == (matrix.shape[0], )
    assert result[2].shape == (matrix.shape[1], matrix.shape[1])

    matrix = np.random.randn(6, 4)
    result = svd(matrix=matrix)
    assert result[0].shape == (matrix.shape[0], matrix.shape[0])
    assert result[1].shape == (matrix.shape[1], )
    assert result[2].shape == (matrix.shape[1], matrix.shape[1])

    rank = 3
    matrix = np.random.randn(6, 4)
    result = svd(matrix=matrix, rank=rank)
    assert result[0].shape == (matrix.shape[0], rank)
    assert result[1].shape == (rank, )
    assert result[2].shape == (rank, matrix.shape[1])

    rank = 3
    matrix = np.random.randn(4, 6)
    result = svd(matrix=matrix, rank=rank)
    assert result[0].shape == (matrix.shape[0], rank)
    assert result[1].shape == (rank, )
    assert result[2].shape == (rank, matrix.shape[1])

