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

        # Dummy test for __repr__
        # TODO: find a proper way to implement this test
        decomposition_interface.dummy_init = "_".join(np.linspace(0, 1, 1000, dtype=str))
        decomposition_interface.dummy_max_iter = 50
        decomposition_interface.dummy_epsilon = 10e-3
        decomposition_interface.dummy_tol = 10e-5
        decomposition_interface.dummy_random_state = None
        decomposition_interface.dummy_verbose = False
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        print(repr(decomposition_interface))
        assert captured_output.getvalue() != ''  # to check that something was actually printed


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
