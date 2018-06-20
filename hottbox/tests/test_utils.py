from .. import version
from ..utils import *


def test_version():
    assert isinstance(version.__version__, str)


def test_quick_tensor():
    shape = (2, 3, 4)
    true_data_arange = np.array([[[0,  1,  2,  3],
                                  [4,  5,  6,  7],
                                  [8,  9, 10, 11]],

                                 [[12, 13, 14, 15],
                                  [16, 17, 18, 19],
                                  [20, 21, 22, 23]]])

    true_mode_names = ["mode-{}".format(i) for i in range(len(shape))]
    true_mode_index = None
    true_state = [[0], [1], [2]]

    tensor = quick_tensor(shape=shape, base='arange')
    np.testing.assert_array_equal(tensor.data, true_data_arange)
    assert tensor.mode_names == true_mode_names
    assert tensor.state.mode_order == true_state
    for mode in tensor.modes:
        assert mode.index is true_mode_index

    tensor = quick_tensor(shape=shape, base='undefined')
    np.testing.assert_array_equal(tensor.data, true_data_arange)

    tensor = quick_tensor(shape=shape, base='randn')
    assert tensor.shape == shape

    tensor = quick_tensor(shape=shape, base='rand')
    assert tensor.shape == shape

    tensor = quick_tensor(shape=shape, base='ones')
    assert tensor.shape == shape
