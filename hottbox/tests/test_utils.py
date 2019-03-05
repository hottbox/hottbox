from ..version import __version__
from ..utils import *
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from ..core._meta import State


def test_version():
    assert isinstance(__version__, str)

def _manualToepTensor():
    tensor = np.zeros(shape=(4,4,3))
    mat_A = genToeplitzMatrix([1,2,3,4],[1,4,3,2])
    mat_B = genToeplitzMatrix([13,5,17,8],[13,18,17,5])
    mat_C = genToeplitzMatrix([0,9,30,11],[0,11,30,9]) 
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C
    return Tensor(array=tensor)


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


def test_sliceT():
    tensor = _manualToepTensor().data
    assert np.array_equal(tensor[0,:,:], sliceT(tensor, 0, 0))
    assert np.array_equal(tensor[:,3,:], sliceT(tensor, 3, 1))
    assert np.array_equal(tensor[:,:,1], sliceT(tensor, 1, 2))


def test_isToepMatrix():
    mat = _manualToepMatrix()
    assert isToepMatrix(mat)

def test_isToepTensor():
    assert isToepTensor(_manualToepTensor())

def test_genToeplitzMatrix():
    mat = genToeplitzMatrix([1,2,3,4,5,6], [1,4,5,6])
    assert np.array_equal(_manualToepMatrix(), mat)

def test_genHankelMatrix():
    mat = genHankelMatrix([1,2,3,4,5,6], [1,4,5,6])
    assert np.array_equal(_manualHankelMatrix(), mat)

def test_quick_tensorcpd():
    """ Tests for `quick_tensorcpd` function """
    full_shape = (5, 6, 7)
    rank = (2,)
    true_mode_names = ["mode-{}".format(i) for i in range(len(full_shape))]
    true_mode_index = None

    base_to_test = ["arange", "randn", "rand", "ones", "undefined"]
    for base in base_to_test:
        tensor_cpd = quick_tensorcpd(full_shape=full_shape, rank=rank, base=base)
        assert isinstance(tensor_cpd, TensorCPD)
        assert tensor_cpd.ft_shape == full_shape
        assert tensor_cpd.rank == rank
        assert tensor_cpd.mode_names == true_mode_names
        for mode in tensor_cpd.modes:
            assert mode.index is true_mode_index


def test_quick_tensortkd():
    """ Tests for `quick_tensortkd` function """
    full_shape = (5, 6, 7)
    rank = (2, 3, 4)
    true_mode_names = ["mode-{}".format(i) for i in range(len(full_shape))]
    true_mode_index = None

    base_to_test = ["arange", "randn", "rand", "ones", "undefined"]
    for base in base_to_test:
        tensor_tkd = quick_tensortkd(full_shape=full_shape, rank=rank, base=base)
        assert isinstance(tensor_tkd, TensorTKD)
        assert tensor_tkd.ft_shape == full_shape
        assert tensor_tkd.rank == rank
        assert tensor_tkd.mode_names == true_mode_names
        for mode in tensor_tkd.modes:
            assert mode.index is true_mode_index


def test_quick_tensortt():
    """ Tests for `quick_tensortt` function """
    full_shape = (5, 6, 7)
    rank = (2, 3)
    true_mode_names = ["mode-{}".format(i) for i in range(len(full_shape))]
    true_mode_index = None

    base_to_test = ["arange", "randn", "rand", "ones", "undefined"]
    for base in base_to_test:
        tensor_tt = quick_tensortt(full_shape=full_shape, rank=rank, base=base)
        assert isinstance(tensor_tt, TensorTT)
        assert tensor_tt.ft_shape == full_shape
        assert tensor_tt.rank == rank
        assert tensor_tt.mode_names == true_mode_names
        for mode in tensor_tt.modes:
            assert mode.index is true_mode_index


def test_quick_tensor():
    """ Tests for `quick_tensor` function """
    shape = (2, 3, 4)
    true_mode_names = ["mode-{}".format(i) for i in range(len(shape))]
    true_mode_index = None
    true_state = State(normal_shape=shape)

    base_to_test = ["arange", "randn", "rand", "ones", "undefined"]
    for base in base_to_test:
        tensor = quick_tensor(shape=shape, base=base)
        assert isinstance(tensor, Tensor)
        assert tensor.shape == shape
        assert tensor.ft_shape == shape
        assert tensor.mode_names == true_mode_names
        assert tensor._state == true_state
        for mode in tensor.modes:
            assert mode.index is true_mode_index
