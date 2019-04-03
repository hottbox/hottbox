from hottbox.utils.generation.basic import *
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


def test_supersymmetric():
    true_shape = (4,4,4)
    tensor = super_symmetric_tensor(true_shape)

    assert true_shape == tensor.shape
    assert isinstance(tensor, Tensor)
    assert is_super_symmetric(tensor)
