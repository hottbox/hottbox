import pytest
import numpy as np
from collections import OrderedDict
from functools import reduce
from ..structures import *


def test_Tensor():
    """ Tests for Tensor class """

    # --------- tests that Tensor can be created only from numpy array
    array = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(TypeError):
        Tensor(array=array)
    with pytest.raises(TypeError):
        Tensor(array=Tensor(array))


    # --------- tests basic properties of a tensor with default names
    true_shape = (2, 4, 8)
    true_size = reduce(lambda x, y: x * y, true_shape)
    true_order = len(true_shape)
    true_data = np.ones(true_size).reshape(true_shape)
    true_default_mode_names = OrderedDict([(0, 'mode-0'),
                                           (1, 'mode-1'),
                                           (2, 'mode-2')])
    tensor = Tensor(array=true_data)
    np.testing.assert_array_equal(tensor.data, true_data)
    assert (tensor.frob_norm == 8.0)
    assert (tensor.shape == true_shape)
    assert (tensor.order == true_order)
    assert (tensor.size == true_size)
    assert (tensor.mode_names == true_default_mode_names)


    # --------- tests on creating a copy of a Tensor object with identical structure and information
    tensor_copy = tensor.copy()
    assert (tensor_copy is not tensor)
    np.testing.assert_array_equal(tensor_copy.data, tensor.data)
    assert (tensor_copy.frob_norm == tensor.frob_norm)
    assert (tensor_copy.shape == tensor.shape)
    assert (tensor_copy.order == tensor.order)
    assert (tensor_copy.size == tensor.size)
    assert (tensor_copy.mode_names == tensor.mode_names)


    # --------- tests for creating a Tensor object with custom mode names
    true_shape = (2, 4, 8)
    true_size = reduce(lambda x, y: x * y, true_shape)
    true_order = len(true_shape)
    true_data = np.ones(true_size).reshape(true_shape)
    true_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(true_order)])
    true_new_mode_names = OrderedDict([(0, 'time'),
                                       (1, 'frequency'),
                                       (2, 'channel')])

    tensor = Tensor(array=true_data, mode_names=true_mode_names)
    assert (tensor.mode_names == true_mode_names)

    tensor.rename_modes(new_mode_names=true_new_mode_names)
    assert (tensor.mode_names == true_new_mode_names)

    # check that renaming also works with normal dict
    true_new_mode_names = {0: 'pixel_x',
                           1: 'pixel_y',
                           2: 'color'}
    tensor.rename_modes(new_mode_names=true_new_mode_names)
    assert (tensor.mode_names == true_new_mode_names)


    # --------- tests that should FAIL for mode names being incorrectly defined at the creation of a Tensor object
    with pytest.raises(TypeError):
        # mode names are not of OrderedDict type
        incorrect_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order)}
        Tensor(array=true_data, mode_names=incorrect_mode_names)

    with pytest.raises(ValueError):
        # not enough mode names
        incorrect_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(true_order - 1)])
        Tensor(array=true_data, mode_names=incorrect_mode_names)

    with pytest.raises(ValueError):
        # too many mode names
        incorrect_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(true_order + 1)])
        Tensor(array=true_data, mode_names=incorrect_mode_names)

    with pytest.raises(TypeError):
        # incorrect type of keys (not integers)
        incorrect_mode_names = OrderedDict([("{}-mode".format(mode), mode) for mode in range(true_order)])
        Tensor(array=true_data, mode_names=incorrect_mode_names)

    with pytest.raises(ValueError):
        # key value exceeds the order of a tensor
        incorrect_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(true_order-2, true_order + 1)])
        Tensor(array=true_data, mode_names=incorrect_mode_names)

    with pytest.raises(ValueError):
        # key value is set to be negative
        incorrect_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(-1, true_order - 1)])
        Tensor(array=true_data, mode_names=incorrect_mode_names)


    # ---------  tests that should FAIL for new mode names being incorrectly defined for renaming
    with pytest.raises(ValueError):
        # too many mode names
        incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order + 1)}
        tensor.rename_modes(new_mode_names=incorrect_new_mode_names)

    with pytest.raises(TypeError):
        # incorrect type of keys (not integers)
        incorrect_new_mode_names = {"{}-mode".format(mode): mode for mode in range(true_order)}
        tensor.rename_modes(new_mode_names=incorrect_new_mode_names)

    with pytest.raises(ValueError):
        # key value exceeds the order of a tensor
        incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order - 2, true_order + 1)}
        tensor.rename_modes(new_mode_names=incorrect_new_mode_names)

    with pytest.raises(ValueError):
        # key value is set to be negative
        incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(-1, true_order - 1)}
        tensor.rename_modes(new_mode_names=incorrect_new_mode_names)




