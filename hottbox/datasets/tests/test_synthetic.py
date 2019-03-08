"""
Tests for the rank estimation module
"""
import pytest
import numpy as np
from ..synthetic import *
from ...core.structures import Tensor
from ...utils.checks import isToepTensor, isToepMatrix


class TestbasicTensor():
    """ Tests for basicTensor class """
    def test_predefined_distr(self):
        true_shape = (4,3,2,3)
        true_distribution = 'normal'
        true_type = 0
        tt = basicTensor(true_shape, true_distribution, true_type)._predefined_distr(true_shape)
        assert tt.shape == true_shape
        assert isinstance(tt, np.ndarray)

    def test_dense(self):
        true_shape = (4,3,2,3)
        true_distribution = 'normal'
        true_type = 0
        tt = basicTensor(true_shape, true_distribution, true_type)
        assert true_shape == tt.dim
        assert true_distribution == tt.distr
        assert true_type == tt.distr_type
        tensor = tt.dense()
        assert isinstance(tensor, Tensor)
        pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
        assert pct_nonzero > 0.8

    def test_sparse(self):
        true_shape = (4,3,2,3)
        true_distribution = 'normal'
        true_type = 0
        tt = basicTensor(true_shape, true_distribution, true_type)
        assert true_shape == tt.dim
        assert true_distribution == tt.distr
        assert true_type == tt.distr_type
        tensor = tt.sparse()
        assert isinstance(tensor, Tensor)
        pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
        assert pct_nonzero < 0.1

    def test_superdiagonal(self):
        true_shape = (4,4,4)
        true_distribution = 'ones'
        true_type = 0
        tt = basicTensor(true_shape, true_distribution, true_type)
        assert true_shape == tt.dim
        assert true_distribution == tt.distr
        assert true_type == tt.distr_type
        tensor = tt.superdiagonal()
        assert isinstance(tensor, Tensor)
        tensor = tensor.data
        trace = 0
        for i in range(true_shape[0]):
            trace += tensor[i,i,i]
        assert trace == true_shape[0]

def test_toeplitzTensor():
    tensor = np.zeros(shape=(4,4,3))
    # Inititalise
    mat_A = genToeplitzMatrix([1,2,3,4],[1,4,3,2])
    mat_B = genToeplitzMatrix([13,5,17,8],[13,18,17,5])
    mat_C = genToeplitzMatrix([0,9,30,11],[0,11,30,9])
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C

    tt = toeplitzTensor((4,4,3), matC=np.array([mat_A, mat_B, mat_C])).data
    assert np.array_equal(tt, tensor)


def test_toeplitzTensorRandom():
    test_tensor = toeplitzTensor((3,3,4), modes=[0,1], random=True)
    assert isToepTensor(test_tensor, modes=[0,1])


