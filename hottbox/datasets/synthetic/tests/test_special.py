import pytest
import numpy as np
from ..special import *
from ....core.structures import Tensor
from ....utils import isToepTensor, isToepMatrix

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


