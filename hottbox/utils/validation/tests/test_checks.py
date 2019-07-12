import numpy as np
from ..checks import is_toeplitz_matrix, is_toeplitz_tensor, is_super_symmetric
from ....core.structures import Tensor
from scipy.linalg import toeplitz as toeplitz_mat


def _manualtoepmatrix():
    mat = np.array([[1, 2, 3, 4, 5, 6],
                    [4, 1, 2, 3, 4, 5],
                    [5, 4, 1, 2, 3, 4],
                    [6, 5, 4, 1, 2, 3]])
    return mat


def _manualtoeptensor():
    tensor = np.zeros(shape=(4,4,3))
    mat_A = toeplitz_mat(r=[1,2,3,4], c=[1,4,3,2])
    mat_B = toeplitz_mat(r=[13,5,17,8],c=[13,18,17,5])
    mat_C = toeplitz_mat(r=[0,9,30,11],c=[0,11,30,9]) 
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C
    return Tensor(array=tensor)


def test_is_toep_matrix():
    mat = _manualtoepmatrix()
    assert is_toeplitz_matrix(mat)


def test_is_toep_tensor():
    assert is_toeplitz_tensor(_manualtoeptensor())


def test_is_super_symmetric():
    tensor = np.array([[[1,2,3], [2,4,5], [3,5,6]],
                       [[2,4,5], [4,7,8], [5,8,9]],
                       [[3,5,6], [5,8,9], [6,9,10]]])
    tensor = Tensor(array=tensor)
    assert is_super_symmetric(tensor)
