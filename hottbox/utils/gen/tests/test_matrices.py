from ..matrices import *
from ....core.structures import Tensor, TensorCPD, TensorTKD, TensorTT

def _manualToepMatrix():
    mat = np.array([[1, 2, 3, 4, 5, 6],
                    [4, 1, 2, 3, 4, 5],
                    [5, 4, 1, 2, 3, 4],
                    [6, 5, 4, 1, 2, 3]])
    return mat

def _manualHankelMatrix():
    mat = np.array([[1, 4, 5, 6, 5, 4],
                    [4, 5, 6, 5, 4, 3],
                    [5, 6, 5, 4, 3, 2],
                    [6, 5, 4, 3, 2, 1]])
    return mat


def test_genToeplitzMatrix():
    mat = genToeplitzMatrix([1,2,3,4,5,6], [1,4,5,6])
    assert np.array_equal(_manualToepMatrix(), mat)

def test_genHankelMatrix():
    mat = genHankelMatrix([1,2,3,4,5,6], [1,4,5,6])
    assert np.array_equal(_manualHankelMatrix(), mat)
