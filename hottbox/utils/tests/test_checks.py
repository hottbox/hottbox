import numpy as np
from functools import reduce
from ..checks import *
from ...core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from ..gen.matrices import genToeplitzMatrix

def _manualtoepmatrix():
    mat = np.array([[1, 2, 3, 4, 5, 6],
                    [4, 1, 2, 3, 4, 5],
                    [5, 4, 1, 2, 3, 4],
                    [6, 5, 4, 1, 2, 3]])
    return mat

def _manualtoeptensor():
    tensor = np.zeros(shape=(4,4,3))
    mat_A = genToeplitzMatrix([1,2,3,4],[1,4,3,2])
    mat_B = genToeplitzMatrix([13,5,17,8],[13,18,17,5])
    mat_C = genToeplitzMatrix([0,9,30,11],[0,11,30,9]) 
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C
    return Tensor(array=tensor)

def test_is_toep_matrix():
    mat = _manualtoepmatrix()
    assert is_toep_matrix(mat)

def test_is_toep_tensor():
    assert is_toep_tensor(_manualtoeptensor())

def test_is_super_symmetric():
    tensor = np.array([[[1,2,3], [2,4,5], [3,5,6]],
                       [[2,4,5], [4,7,8], [5,8,9]],
                       [[3,5,6], [5,8,9], [6,9,10]]])
    tensor = Tensor(array=tensor)
    assert is_super_symmetric(tensor)
