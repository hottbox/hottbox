import numpy as np
from functools import reduce
from ..checks import *
from ...core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from ..gen.matrices import genToeplitzMatrix

def _manualToepMatrix():
    mat = np.array([[1, 2, 3, 4, 5, 6],
                    [4, 1, 2, 3, 4, 5],
                    [5, 4, 1, 2, 3, 4],
                    [6, 5, 4, 1, 2, 3]])
    return mat

def _manualToepTensor():
    tensor = np.zeros(shape=(4,4,3))
    mat_A = genToeplitzMatrix([1,2,3,4],[1,4,3,2])
    mat_B = genToeplitzMatrix([13,5,17,8],[13,18,17,5])
    mat_C = genToeplitzMatrix([0,9,30,11],[0,11,30,9]) 
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C
    return Tensor(array=tensor)

def test_isToepMatrix():
    mat = _manualToepMatrix()
    assert isToepMatrix(mat)

def test_isToepTensor():
    assert isToepTensor(_manualToepTensor())
