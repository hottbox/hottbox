from ...version import __version__
from ..utils import *
from ...core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from ...core._meta import State
from ..gen.matrices import genToeplitzMatrix

def _manualToepTensor():
    tensor = np.zeros(shape=(4,4,3))
    mat_A = genToeplitzMatrix([1,2,3,4],[1,4,3,2])
    mat_B = genToeplitzMatrix([13,5,17,8],[13,18,17,5])
    mat_C = genToeplitzMatrix([0,9,30,11],[0,11,30,9]) 
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C
    return Tensor(array=tensor)

def test_version():
    assert isinstance(__version__, str)

def test_sliceT():
    tensor = _manualToepTensor().data
    assert np.array_equal(tensor[0,:,:], sliceT(tensor, 0, 0))
    assert np.array_equal(tensor[:,3,:], sliceT(tensor, 3, 1))
    assert np.array_equal(tensor[:,:,1], sliceT(tensor, 1, 2))
