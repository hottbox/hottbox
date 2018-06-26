from ..version import __version__
from ..utils import *
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT
from ..core._meta import State


def test_version():
    assert isinstance(__version__, str)


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
