import pytest
import numpy as np
from hottbox.utils.generation.basic import dense_tensor, sparse_tensor, super_diagonal_tensor, \
    super_diag_tensor, super_symmetric_tensor
from hottbox.core.structures import Tensor
from hottbox.utils.validation.checks import is_super_symmetric


def test_dense():
    # TODO: test distribution
    true_shape = (4,3,2,3)
    true_distribution = 'normal'
    true_type = 0
    tensor = dense_tensor(true_shape, true_distribution, true_type)
    assert isinstance(tensor, Tensor)
    assert true_shape == tensor.shape
    pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
    assert pct_nonzero > 0.8


def test_sparse():
    true_shape = (4,3,2,3)
    true_distribution = 'normal'
    true_type = 0
    tensor = sparse_tensor(true_shape, true_distribution, true_type)
    assert isinstance(tensor, Tensor)
    assert true_shape == tensor.shape
    pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
    assert pct_nonzero < 0.08 and pct_nonzero > 0.02


def test_superdiagonal():
    true_shape = (4,4,4)
    true_distribution = 'ones'
    true_type = 0
    tensor = super_diagonal_tensor(true_shape, true_distribution)
    assert true_shape == tensor.shape
    assert isinstance(tensor, Tensor)
    tensor = tensor.data
    trace = 0
    for i in range(true_shape[0]):
        trace += tensor[i,i,i]
    assert trace == true_shape[0]


def test_super_diag_tensor():
    """ Tests for creating super-diagonal tensor"""
    order = 3
    rank = 2
    correct_shape = (rank, ) * order
    true_default_data = np.array([[[1., 0.],
                                   [0., 0.]],

                                  [[0., 0.],
                                   [0., 1.]]])
    true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
    correct_values = np.arange(rank)
    true_data = np.array([[[0., 0.],
                           [0., 0.]],

                          [[0., 0.],
                           [0., 1.]]])

    # ------ tests for default super diagonal tensor
    tensor = super_diag_tensor(correct_shape)
    assert isinstance(tensor, Tensor)
    np.testing.assert_array_equal(tensor.data, true_default_data)
    assert (tensor.mode_names == true_default_mode_names)

    # ------ tests for super diagonal tensor with custom values on the main diagonal
    tensor = super_diag_tensor(correct_shape, values=correct_values)
    assert isinstance(tensor, Tensor)
    np.testing.assert_array_equal(tensor.data, true_data)
    assert (tensor.mode_names == true_default_mode_names)

    # ------ tests that should Fail

    with pytest.raises(TypeError):
        # shape should be passed as tuple
        super_diag_tensor(shape=list(correct_shape))

    with pytest.raises(ValueError):
        # all values in shape should be the same
        incorrect_shape = [rank] * order
        incorrect_shape[1] = order+1
        super_diag_tensor(shape=tuple(incorrect_shape))

    with pytest.raises(ValueError):
        # values should be an one dimensional numpy array
        incorrect_values = np.ones([rank, rank])
        super_diag_tensor(shape=correct_shape, values=incorrect_values)

    with pytest.raises(ValueError):
        # too many values for the specified shape
        incorrect_values = np.ones(correct_shape[0]+1)
        super_diag_tensor(shape=correct_shape, values=incorrect_values)

    with pytest.raises(TypeError):
        # values should be a numpy array
        incorrect_values = [1] * correct_shape[0]
        super_diag_tensor(shape=correct_shape, values=incorrect_values)


def test_supersymmetric():
    true_shape = (4,4,4)
    tensor = super_symmetric_tensor(true_shape)

    assert true_shape == tensor.shape
    assert isinstance(tensor, Tensor)
    assert is_super_symmetric(tensor)
