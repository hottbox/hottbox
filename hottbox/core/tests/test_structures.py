import pytest
import sys
import io
import numpy as np
from functools import reduce
from ..structures import *
from ..operations import unfold, kolda_unfold
from .._meta import State
from ...errors import TensorModeError, TensorShapeError, TensorStateError, TensorTopologyError, ModeError, StateError


# TODO: find a better way to test the methods that only prints and for __repr__ and __str__
class TestTensor:
    """ Tests for Tensor class """

    def test_init(self):
        """ Tests for Tensor object creation """
        # ------ tests for basic properties of a tensor with default mode names
        true_shape = (2, 4, 8)
        true_size = reduce(lambda x, y: x * y, true_shape)
        true_order = len(true_shape)
        true_data = np.ones(true_size).reshape(true_shape)
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
        true_default_state = State(normal_shape=true_shape)
        tensor = Tensor(array=true_data)
        np.testing.assert_array_equal(tensor.data, true_data)
        assert (tensor.frob_norm == 8.0)
        assert (tensor.shape == true_shape)
        assert (tensor.ft_shape == true_shape)
        assert (tensor.order == true_order)
        assert (tensor.size == true_size)
        assert (tensor.mode_names == true_default_mode_names)
        assert (tensor._state == true_default_state)
        assert (tensor._data is not true_data)          # check that is not a reference

        # ------ tests for creating a Tensor object with custom mode names
        true_custom_mode_names = ['time', 'frequency', 'channel']
        tensor = Tensor(array=true_data, mode_names=true_custom_mode_names)
        assert (tensor.mode_names == true_custom_mode_names)

        # ------ tests for creating a Tensor object in custom state
        I, J, K = 2, 4, 8
        true_data = np.ones(I * J * K).reshape(I, J*K)
        true_ft_shape = (I, J, K)
        true_mode_order = ([0], [1, 2])
        true_mode_names = ["mode-0", "mode-1_mode-2"]
        custom_state = dict(normal_shape=true_ft_shape,
                            mode_order=true_mode_order,
                            rtype="T")
        tensor = Tensor(array=true_data, custom_state=custom_state)
        assert (tensor.ft_shape == true_ft_shape)
        assert (tensor.mode_names == true_mode_names)

        # ------ tests for creating a Tensor object with in custom state and with custom mode names
        I, J, K = 2, 4, 8
        true_data = np.ones(I * J * K).reshape(I, J * K)
        true_ft_shape = (I, J, K)
        true_mode_order = ([0], [1, 2])
        custom_mode_names = ["time", "frequency", "channel"]
        true_custom_mode_names = ["time", "frequency_channel"]
        custom_state = dict(normal_shape=true_ft_shape,
                            mode_order=true_mode_order,
                            rtype="T")
        tensor = Tensor(array=true_data, custom_state=custom_state, mode_names=custom_mode_names)
        assert (tensor.ft_shape == true_ft_shape)
        assert (tensor.mode_names == true_custom_mode_names)

    def test_init_fail(self):
        """ Tests for incorrect input data for the Tensor constructor """
        # ------ the following tests should FAIL
        correct_shape = (2, 4, 8)
        size = reduce(lambda x, y: x * y, correct_shape)
        order = len(correct_shape)
        correct_data = np.ones(size).reshape(correct_shape)

        # ------ tests that Tensor object can be created only from numpy array
        # can not create from list
        with pytest.raises(TypeError):
            incorrect_data = [[1, 2, 3], [4, 5, 6]]
            Tensor(array=incorrect_data)

        # can not create from another Tensor
        with pytest.raises(TypeError):
            incorrect_data = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
            Tensor(array=incorrect_data)

        # ------ tests for custom mode names being incorrectly defined
        # mode names are not of list type
        with pytest.raises(ModeError):
            incorrect_mode_names = {mode: "{}-mode".format(mode) for mode in range(order)}
            Tensor(array=correct_data, mode_names=incorrect_mode_names)

        # not enough mode names
        with pytest.raises(ModeError):
            incorrect_mode_names = ["{}-mode".format(mode) for mode in range(order - 1)]
            Tensor(array=correct_data, mode_names=incorrect_mode_names)

        # too many mode names
        with pytest.raises(ModeError):
            incorrect_mode_names = ["{}-mode".format(mode) for mode in range(order + 1)]
            Tensor(array=correct_data, mode_names=incorrect_mode_names)

        # all mode names should be strings
        with pytest.raises(ModeError):
            incorrect_mode_names = ["{}-mode".format(mode) for mode in range(order)]
            incorrect_mode_names[0] = 0
            Tensor(array=correct_data, mode_names=incorrect_mode_names)

        # ------ tests for custom state being incorrectly defined
        # custom state should be passed as a dict
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            correct_mode_order = ([0], [1], [2])
            correct_rtype = "T"

            incorrect_custom_state = [correct_normal_shape,
                                      correct_mode_order,
                                      correct_rtype]
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # custom state not fully defined
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            correct_mode_order = ([0], [1], [2])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=correct_mode_order)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # normal shape of custom state should be a tuple
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            incorrect_normal_shape = [I, J, K]
            correct_mode_order = ([0], [1], [2])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=incorrect_normal_shape,
                                          mode_order=correct_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # normal shape of custom state is inconsistent with the shape of provided data
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            incorrect_normal_shape = (I+1, J, K)
            correct_mode_order = ([0], [1], [2])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=incorrect_normal_shape,
                                          mode_order=correct_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # mode order of custom state should be a !! TUPLE !! of lists
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            incorrect_mode_order = [[0], [1], [2]]
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=incorrect_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # mode order of custom state should be a tuple of !! LISTS !!
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            incorrect_mode_order = (0, 1, 2)
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=incorrect_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # number of list in mode order should correspond to the number of dimensions of provided data
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            incorrect_mode_order = ([0], [1, 2])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=incorrect_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J*K)
            correct_normal_shape = (I, J, K)
            incorrect_mode_order = ([0], [1], [2])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=incorrect_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # length of mode order of custom state is inconsistent with the normal shape
        with pytest.raises(StateError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            incorrect_mode_order = ([0], [1], [2, 3])
            correct_rtype = "T"

            incorrect_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=incorrect_mode_order,
                                          rtype=correct_rtype)
            Tensor(array=correct_data, custom_state=incorrect_custom_state)

        # length of normal shape of custom state is inconsistent with the length of provided mode names
        with pytest.raises(ModeError):
            I, J, K = 2, 4, 8
            correct_data = np.ones(I * J * K).reshape(I, J, K)
            correct_normal_shape = (I, J, K)
            correct_mode_order = ([0], [1], [2])
            correct_rtype = "T"
            correct_custom_state = dict(normal_shape=correct_normal_shape,
                                          mode_order=correct_mode_order,
                                          rtype=correct_rtype)
            incorrect_mode_names = ["frequency", "time"]
            Tensor(array=correct_data, custom_state=correct_custom_state, mode_names=incorrect_mode_names)

    def test_equal(self):
        """ Test for tensors being equal """
        shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, shape)
        data_1 = np.ones(size).reshape(shape)
        data_2 = np.ones(size).reshape(shape)
        data_3 = np.arange(size).reshape(shape)
        init_names = ["country", "year", "month"]
        new_mode_names = {i: "{}".format(init_names[i]) for i in range(len(shape))}
        new_mode_index = {i: ["index" for _ in range(shape[i])] for i in range(len(shape))}

        tensor_1 = Tensor(data_1)
        tensor_2 = Tensor(data_2)
        assert tensor_1 == tensor_2

        tensor_1 = Tensor(array=data_1, mode_names=init_names)
        tensor_2 = Tensor(array=data_2, mode_names=init_names)
        assert tensor_1 == tensor_2

        tensor_1 = Tensor(array=data_1).set_mode_index(mode_index=new_mode_index)
        tensor_2 = Tensor(array=data_2).set_mode_index(mode_index=new_mode_index)
        assert tensor_1 == tensor_2

        tensor_1 = Tensor(array=data_1, mode_names=init_names)
        tensor_2 = Tensor(array=data_2)
        assert tensor_1 != tensor_2
        tensor_2.set_mode_names(mode_names=new_mode_names)
        assert tensor_1 == tensor_2

        tensor_1 = Tensor(array=data_1)
        tensor_2 = Tensor(array=data_3)
        assert tensor_1 != tensor_2

        tensor_1 = Tensor(array=data_1)
        tensor_2 = Tensor(array=data_2).unfold(mode=0, inplace=True)
        assert tensor_1 != tensor_2

        tensor_1 = Tensor(array=data_1, mode_names=init_names)
        tensor_2 = Tensor(array=data_2)
        assert tensor_1 != tensor_2

        tensor_1 = Tensor(array=data_1).set_mode_index(mode_index=new_mode_index)
        tensor_2 = Tensor(array=data_2)
        assert tensor_1 != tensor_2

        assert tensor_1 != data_1

    def test_addition(self):
        """ Test for summation of two tensors """
        shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, shape)
        data_1 = np.ones(size).reshape(shape)
        data_2 = np.arange(size).reshape(shape)
        data_res = data_1 + data_2

        tensor_1 = Tensor(array=data_1)
        tensor_2 = Tensor(array=data_2)
        tensor_res = Tensor(array=data_res)
        tensor = tensor_1 + tensor_2
        assert tensor_res == tensor

        tensor_1 = Tensor(array=data_1, mode_names=["country", "year", "month"])
        tensor_2 = Tensor(array=data_2)
        tensor_res = Tensor(array=data_res)
        tensor = tensor_1 + tensor_2
        assert tensor_res == tensor

        #---- Tests that should fail
        with pytest.raises(TypeError):
            assert Tensor(data_1) + data_1

        with pytest.raises(TensorStateError):
            tensor_1 = Tensor(array=data_1)
            tensor_2 = Tensor(array=data_2).unfold(mode=0, inplace=True)
            assert tensor_1 + tensor_2

        with pytest.raises(TensorModeError):
            mode_index = {0: ["idx1", "idx2"]}
            tensor_1 = Tensor(array=data_1)
            tensor_2 = Tensor(array=data_2).set_mode_index(mode_index=mode_index)
            assert tensor_1 + tensor_2

        with pytest.raises(TensorShapeError):
            data_2 = np.arange(2*2).reshape(2,2)
            tensor_1 = Tensor(array=data_1)
            tensor_2 = Tensor(array=data_2)
            assert tensor_1 + tensor_2

    def test_repr(self):
        array = np.arange(2*3*4).reshape(2,3,4)
        tensor = Tensor(array=array)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output     # and redirect stdout.
        print(repr(tensor))
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_show_state(self):
        """ Tests for `show_state` method """
        array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        tensor = Tensor(array=array)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        tensor.show_state()
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_copy(self):
        """ Tests for creation a copy of a Tensor object """
        data = np.arange(24).reshape(2, 3, 4)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"],}
        tensor = Tensor(data, mode_names=['pixel_x', 'pixel_y', 'color']).set_mode_index(mode_index=mode_index)
        tensor_copy = tensor.copy()
        assert tensor_copy == tensor
        assert (tensor_copy is not tensor)
        assert (tensor_copy._data is not tensor._data)
        assert (tensor_copy._modes is not tensor._modes)
        assert (tensor_copy._state is not tensor._state)

    def test_reset_meta(self):
        """ Tests for `reset_meta` method """
        shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, shape)
        data = np.ones(size).reshape(shape)
        init_names = ["country", "year", "month"]
        mode_index = {i: ["index" for _ in range(shape[i])] for i in range(len(shape))}

        tensor = Tensor(array=data, mode_names=init_names).set_mode_index(mode_index=mode_index)
        tensor.reset_meta()
        tensor_true_result = Tensor(array=data)
        assert tensor == tensor_true_result

    def test_set_mode_names(self):
        """ Tests for renaming modes """
        true_shape = (2, 4, 8)
        true_size = reduce(lambda x, y: x * y, true_shape)
        true_order = len(true_shape)
        true_data = np.ones(true_size).reshape(true_shape)
        true_new_mode_names = {0: 'pixel-x',
                               1: 'pixel-y',
                               2: 'color'
                               }
        tensor = Tensor(array=true_data)
        tensor.set_mode_names(true_new_mode_names)
        assert tensor.mode_names == list(true_new_mode_names.values())

        # ------ tests that should FAIL for new mode names being incorrectly defined for renaming
        with pytest.raises(ModeError):
            # too many mode names
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order + 1)}
            tensor.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # incorrect type of keys (not integers)
            incorrect_new_mode_names = {"{}-mode".format(mode): mode for mode in range(true_order)}
            tensor.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order - 2, true_order + 1)}
            tensor.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # key value is set to be negative
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(-1, true_order - 1)}
            tensor.set_mode_names(mode_names=incorrect_new_mode_names)

    def test_reset_mode_name(self):
        """ Tests for `reset_mode_name` method """
        shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, shape)
        data = np.ones(size).reshape(shape)
        init_names = ["country", "year", "month"]
        mode_index = {i: ["index" for _ in range(shape[i])] for i in range(len(shape))}

        tensor = Tensor(array=data, mode_names=init_names).set_mode_index(mode_index=mode_index)
        tensor.reset_mode_name()
        tensor_true_result = Tensor(array=data).set_mode_index(mode_index=mode_index)
        assert tensor == tensor_true_result

        tensor = Tensor(array=data, mode_names=init_names)
        tensor.reset_mode_name(mode=0)
        init_names[0] = "mode-0"
        tensor_true_result = Tensor(array=data, mode_names=init_names)
        assert tensor == tensor_true_result

    def test_set_mode_index(self):
        """ Tests for `set_mode_index` method """
        shape = (2, 2, 2)
        true_order = len(shape)
        size = reduce(lambda x, y: x * y, shape)
        data = np.ones(size).reshape(shape)
        tensor = Tensor(array=data)

        # ------ tests that should FAIL for new mode index being incorrectly defined for renaming
        with pytest.raises(ModeError):
            # too many lists of indices provided
            mode_index = {i: ["index"] for i in range(len(shape)+1)}
            tensor.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # incorrect type of keys (not integers)
            mode_index = {"index-name": mode for mode in range(true_order)}
            tensor.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            wrong_key = true_order + 1
            mode_index = {wrong_key : ["idx"]}
            tensor.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            wrong_key = -1
            mode_index = {wrong_key : ["idx"]}
            tensor.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # not enough indices for the length of the mode
            mode_index = {0: ["idx"]}
            tensor.set_mode_index(mode_index=mode_index)

    def test_reset_mode_index(self):
        """ Tests for `reset_mode_index` method """
        shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, shape)
        data = np.ones(size).reshape(shape)
        init_names = ["country", "year", "month"]
        mode_index = {i: ["index" for _ in range(shape[i])] for i in range(len(shape))}

        tensor = Tensor(array=data, mode_names=init_names).set_mode_index(mode_index=mode_index)
        tensor.reset_mode_index()
        tensor_true_result = Tensor(array=data, mode_names=init_names)
        assert tensor == tensor_true_result

    def test_describe(self):
        """ Tests for describe function of a Tensor object """
        array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        tensor = Tensor(array=array)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        tensor.describe()
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_unfold_fold(self):
        """ Tests for folding and unfolding of a Tensor object """
        shape = (2, 3, 4)
        size = reduce(lambda x, y: x * y, shape)

        orig_data = np.arange(size).reshape(shape)
        unfolded_data = [unfold(tensor=orig_data, mode=mode) for mode in range(len(shape))]
        kolda_unfolded_data = [kolda_unfold(tensor=orig_data, mode=mode) for mode in range(len(shape))]

        orig_mode_names = ['time', 'frequency', 'person']
        unfolded_mode_names = [
            ['time', 'frequency_person'],
            ['frequency', 'time_person'],
            ['person', 'time_frequency']
        ]

        orig_state = State(normal_shape=shape)
        unfolded_state = [
            State(normal_shape=shape, rtype="T", mode_order=([0], [1, 2])),
            State(normal_shape=shape, rtype="T", mode_order=([1], [0, 2])),
            State(normal_shape=shape, rtype="T", mode_order=([2], [0, 1])),
        ]
        unfolded_state_kolda = [
            State(normal_shape=shape, rtype="K", mode_order=([0], [1, 2])),
            State(normal_shape=shape, rtype="K", mode_order=([1], [0, 2])),
            State(normal_shape=shape, rtype="K", mode_order=([2], [0, 1])),
        ]

        tensor = Tensor(array=orig_data, mode_names=orig_mode_names)

        # --------- tests for unfolding and folding INPLACE=TRUE
        for mode in range(len(shape)):
            tensor.unfold(mode=mode, inplace=True)
            assert tensor._state == unfolded_state[mode]
            np.testing.assert_array_equal(tensor.data, unfolded_data[mode])
            assert (tensor.mode_names == unfolded_mode_names[mode])

            tensor.fold(inplace=True)
            assert tensor._state == orig_state
            np.testing.assert_array_equal(tensor.data, orig_data)
            assert (tensor.mode_names == orig_mode_names)

        # --------- tests for kolda unfolding and folding INPLACE=TRUE
        for mode in range(len(shape)):
            tensor.unfold(mode=mode, rtype="K", inplace=True)
            assert tensor._state == unfolded_state_kolda[mode]
            np.testing.assert_array_equal(tensor.data, kolda_unfolded_data[mode])
            assert (tensor.mode_names == unfolded_mode_names[mode])

            tensor.fold(inplace=True)
            assert tensor._state == orig_state
            np.testing.assert_array_equal(tensor.data, orig_data)
            assert (tensor.mode_names == orig_mode_names)

        # --------- tests for unfolding and folding INPLACE=FALSE
        for mode in range(len(shape)):
            tensor_unfolded = tensor.unfold(mode=mode, inplace=False)
            tensor_folded = tensor_unfolded.fold(inplace=False)

            assert tensor_unfolded is not tensor
            assert tensor_folded is not tensor_unfolded

            assert tensor._state == orig_state
            assert (tensor.mode_names == orig_mode_names)
            np.testing.assert_array_equal(tensor.data, orig_data)

            assert tensor_unfolded._state == unfolded_state[mode]
            assert (tensor_unfolded.mode_names == unfolded_mode_names[mode])
            np.testing.assert_array_equal(tensor_unfolded.data, unfolded_data[mode])

            assert tensor_folded._state == orig_state
            assert (tensor_folded.mode_names == orig_mode_names)
            np.testing.assert_array_equal(tensor_folded.data, orig_data)

        # --------- tests that should fail
        with pytest.raises(ValueError):
            # Unknown convention for unfolding
            Tensor(array=orig_data).unfold(mode=0, rtype="dummy", inplace=True)

        # Tests for checking normal state of a tensor
        with pytest.raises(TensorStateError):
            # Should not unfold several times in a row
            Tensor(array=orig_data).unfold(mode=0, inplace=True).unfold(mode=0, inplace=True)

        with pytest.raises(TensorStateError):
            # Should not fold if it wasn't unfolded before
            Tensor(array=orig_data).fold(inplace=True)

    def test_vectorise(self):
        """ Tests for `vectorise` method """
        shape = (2, 3, 4)
        size = reduce(lambda x, y: x * y, shape)

        orig_data = np.arange(size).reshape(shape)

        orig_mode_names = ['time', 'frequency', 'person']
        vectorised_mode_names = ['time_frequency_person']

        orig_state = State(normal_shape=shape)
        vectorised_state = State(normal_shape=shape, rtype="T", mode_order=([0, 1, 2], ))
        vectorised_state_kolda = State(normal_shape=shape, rtype="K", mode_order=([0, 1, 2], ))

        tensor = Tensor(array=orig_data, mode_names=orig_mode_names)

        # --------- tests for vectorising and folding INPLACE=TRUE
        tensor.vectorise(inplace=True)
        assert tensor._state == vectorised_state
        assert (tensor.mode_names == vectorised_mode_names)
        tensor.fold(inplace=True)
        assert tensor._state == orig_state
        np.testing.assert_array_equal(tensor.data, orig_data)
        assert (tensor.mode_names == orig_mode_names)

        tensor.vectorise(rtype="K", inplace=True)
        assert tensor._state == vectorised_state_kolda
        assert (tensor.mode_names == vectorised_mode_names)
        tensor.fold(inplace=True)
        assert tensor._state == orig_state
        np.testing.assert_array_equal(tensor.data, orig_data)
        assert (tensor.mode_names == orig_mode_names)

        # --------- tests for vectorising and folding INPLACE=FALSE
        tensor_vectorised = tensor.vectorise(inplace=False)
        tensor_folded = tensor_vectorised.fold(inplace=False)

        assert tensor_vectorised is not tensor
        assert tensor_folded is not tensor_vectorised

        assert tensor._state == orig_state
        assert (tensor.mode_names == orig_mode_names)
        np.testing.assert_array_equal(tensor.data, orig_data)

        assert tensor_vectorised._state == vectorised_state
        assert (tensor_vectorised.mode_names == vectorised_mode_names)

        assert tensor_folded._state == orig_state
        assert (tensor_folded.mode_names == orig_mode_names)
        np.testing.assert_array_equal(tensor_folded.data, orig_data)

        # --------- tests that should fail
        with pytest.raises(ValueError):
            # Unknown convention for vectorisation
            Tensor(array=orig_data).vectorise(rtype="dummy", inplace=True)

        # Tests for checking normal state of a tensor
        with pytest.raises(TensorStateError):
            # Should vectorise a tensor only if it was in normal state
            Tensor(array=orig_data).unfold(mode=0, inplace=True).vectorise(inplace=True)

    def test_mode_n_product(self):
        """ Tests for mode-n product on an object of Tensor class """
        I, J, K = 5, 6, 7
        I_new, J_new, K_new = 2, 3, 4
        array_3d = np.arange(I * J * K).reshape((I, J, K))
        A = np.arange(I_new * I).reshape(I_new, I)
        B = np.arange(J_new * J).reshape(J_new, J)
        C = np.arange(K_new * K).reshape(K_new, K)
        res_0 = mode_n_product(tensor=array_3d, matrix=A, mode=0)
        res_1 = mode_n_product(tensor=res_0, matrix=B, mode=1)
        res_1 = mode_n_product(tensor=res_1, matrix=C, mode=2)

        tensor = Tensor(array=array_3d)
        tensor.mode_n_product(A, 0)
        np.testing.assert_array_equal(tensor.data, res_0)

        # ------  test for chaining methods
        tensor = Tensor(array=array_3d)
        tensor.mode_n_product(A, 0).mode_n_product(B, 1).mode_n_product(C, 2)
        np.testing.assert_array_equal(tensor.data, res_1)

        # ------  test that chaining order doesn't matter
        tensor = Tensor(array=array_3d)
        tensor.mode_n_product(C, 2).mode_n_product(B, 1).mode_n_product(A, 0)
        np.testing.assert_array_equal(tensor.data, res_1)

        # ------ test for inplace=False
        orig_dim = (5, 6, 7)
        new_dim = [2, 3, 4]
        size = reduce(lambda x, y: x * y, orig_dim)
        array_3d = np.arange(size).reshape(orig_dim)
        tensor = Tensor(array_3d)
        matrix_list = [np.arange(new_dim[i] * orig_dim[i]).reshape(new_dim[i], orig_dim[i]) for i in range(len(new_dim))]
        for mode in range(tensor.order):
            true_res = mode_n_product(tensor=array_3d, matrix=matrix_list[mode], mode=mode)
            tensor_res = tensor.mode_n_product(matrix_list[mode], mode, inplace=False)

            assert (tensor_res is not tensor)
            assert (tensor_res.ft_shape is not tensor.ft_shape)
            np.testing.assert_array_equal(tensor_res.data, true_res)
            # check that the original tensor object has not been modified
            np.testing.assert_array_equal(tensor.data, array_3d)

        # check that mode_n_product can be performed only on a tensor in normal state
        with pytest.raises(TensorStateError):
            tensor = Tensor(array=array_3d).unfold(mode=0, inplace=True)
            matrix = np.arange(2)
            tensor.mode_n_product(matrix, mode=0)

        # ------  test for changing mode_names correctly
        orig_dim = (5, 6, 7)
        new_dim = [2, 3, 4]
        size = reduce(lambda x, y: x * y, orig_dim)
        array_3d = np.arange(size).reshape(orig_dim)
        orig_names = ['country', 'model', 'year']

        # check that names have not been changed when multiply with numpy array
        for i, mode in enumerate(new_dim):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = np.arange(mode * orig_dim[i]).reshape(mode, orig_dim[i])
            tensor.mode_n_product(matrix, mode=i)
            assert (tensor.mode_names == orig_names)

        # check that names have not been changed when multiply with matrix as a Tensor object with default names
        for mode in range(len(new_dim)):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = Tensor(np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode]))
            tensor.mode_n_product(matrix, mode=mode)
            assert (tensor.mode_names == orig_names)

        # check that name of the correct mode has been changed when multiplied with numpy array and specifying new_name
        new_name = 'age'
        for mode in range(len(new_dim)):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode])
            tensor.mode_n_product(matrix, mode=mode, new_name=new_name)
            new_true_names = orig_names.copy()
            new_true_names[mode] = new_name
            assert (tensor.mode_names == new_true_names)

        # check that name of the correct mode has been changed when multiplied with a matrix as a Tensor object
        new_matrix_name = {0: 'age'}
        for mode in range(len(new_dim)):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = Tensor(np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode]))
            matrix.set_mode_names(mode_names=new_matrix_name)
            tensor.mode_n_product(matrix, mode=mode)
            new_true_names = orig_names.copy()
            new_true_names[mode] = new_matrix_name[0]
            assert (tensor.mode_names == new_true_names)

        # check that you cannot use matrix of Tensor class and specify new name at the same time
        with pytest.raises(ModeError):
            mode = 1
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = Tensor(np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode]))
            new_name = 'age'
            tensor.mode_n_product(matrix, mode=mode, new_name=new_name)

        # check that new_name should be of string type
        with pytest.raises(ModeError):
            mode = 1
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode])
            new_name = 5
            tensor.mode_n_product(matrix, mode=mode, new_name=new_name)


class TestBaseTensorTD:
    """ Tests for the BaseTensorTD as an interface"""

    def test_init(self):
        tensor_interface = BaseTensorTD()
        with pytest.raises(NotImplementedError):
            tensor_interface._validate_init_data()
        with pytest.raises(NotImplementedError):
            tensor_interface.copy()
        with pytest.raises(NotImplementedError):
            tensor_interface.modes
        with pytest.raises(NotImplementedError):
            tensor_interface.order
        with pytest.raises(NotImplementedError):
            tensor_interface.rank
        with pytest.raises(NotImplementedError):
            tensor_interface.size
        with pytest.raises(NotImplementedError):
            tensor_interface.frob_norm
        with pytest.raises(NotImplementedError):
            tensor_interface.unfold()
        with pytest.raises(NotImplementedError):
            tensor_interface.fold()
        with pytest.raises(NotImplementedError):
            tensor_interface.mode_n_product()
        with pytest.raises(NotImplementedError):
            tensor_interface.reconstruct()


class TestTensorCPD:
    """ Tests for the TensorCPD class """

    def test_init(self):
        """ Tests for the TensorCPD constructor """
        ft_shape = (3, 4, 5)    # define shape of the tensor in full form
        R = 2                   # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        true_orig_fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]
        true_mode_names = ["mode-0", "mode-1", "mode-2"]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        assert isinstance(tensor_cpd.fmat, list)
        assert tensor_cpd.order == len(fmat_list)
        assert isinstance(tensor_cpd.rank, tuple)
        assert tensor_cpd.rank == (R,)
        assert isinstance(tensor_cpd._core_values, np.ndarray)
        np.testing.assert_array_equal(tensor_cpd._core_values, core_values)
        assert tensor_cpd._core_values is not core_values
        assert tensor_cpd.mode_names == true_mode_names
        assert tensor_cpd.ft_shape == ft_shape

        # ------ tests for factor matrices
        for mode, fmat in enumerate(tensor_cpd.fmat):
            # check that values are the same but there are not references
            np.testing.assert_array_equal(fmat, fmat_list[mode])
            assert fmat is not fmat_list[mode]

            # check that changes to the matrices  have no affect on the TensorCPD
            # (double check for not being references)
            fmat_list[mode] = fmat_list[mode] * 2
            np.testing.assert_array_equal(fmat, true_orig_fmat_list[mode])
            assert fmat is not true_orig_fmat_list[mode]

        # ------ tests for core
        true_core = np.array([[[1., 0.],
                               [0., 0.]],

                              [[0., 0.],
                               [0., 1.]]]
                             )
        assert isinstance(tensor_cpd.core, Tensor)
        np.testing.assert_array_equal(tensor_cpd.core.data, true_core)

    def test_init_fail(self):
        """ Tests for incorrect input data for the TensorCPD constructor """
        # ------ the following tests should FAIL

        ft_shape = (3, 4, 5)  # define shape of the tensor in full form
        R = 2  # define Kryskal rank of a tensor in CP form
        correct_core_values = np.ones(R)
        correct_fmat = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]

        # core_values should be in form of numpy array
        with pytest.raises(TypeError):
            incorrect_core_values = list(correct_core_values)
            TensorCPD(fmat=correct_fmat, core_values=incorrect_core_values)

        # factor matrices should be passed as a list
        with pytest.raises(TypeError):
            incorrect_fmat = tuple(correct_fmat)
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

        # all factor matrices should be in form of numpy array
        with pytest.raises(TypeError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = list(incorrect_fmat[0])
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

        # all factor matrices should be a 2-dimensional numpy array
        with pytest.raises(TensorTopologyError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = np.ones([2, 2, 2])
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

        # too many (or not enough) `core_values` for `fmat`
        with pytest.raises(TensorTopologyError):
            incorrect_core_values = np.ones(correct_core_values.size + 1)
            TensorCPD(fmat=correct_fmat, core_values=incorrect_core_values)

        # dimension all factor matrices should have the same number of columns
        with pytest.raises(TensorTopologyError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = incorrect_fmat[0].T
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

    def test_equal(self):
        """ Test for tensors in kruskal form being equal """
        ft_shape_1 = (3, 4, 5)
        R_1 = 2
        mode_names_1 = ["frequency", "time", "channel"]
        new_mode_names = {i: "{}".format(mode_names_1[i]) for i in range(len(ft_shape_1))}
        new_mode_index = {i: ["index" for _ in range(ft_shape_1[i])] for i in range(len(ft_shape_1))}

        ft_shape_2 = ft_shape_1
        R_2 = R_1
        mode_names_2 = [name for name in mode_names_1]

        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.arange(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        assert t_cpd_1 == t_cpd_2

        t_cpd_1.set_mode_names(mode_names=new_mode_names)
        t_cpd_2.set_mode_names(mode_names=new_mode_names)
        assert t_cpd_1 == t_cpd_2

        t_cpd_1.set_mode_index(mode_index=new_mode_index)
        t_cpd_2.set_mode_index(mode_index=new_mode_index)
        assert t_cpd_1 == t_cpd_2

        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1), mode_names=mode_names_1)
        t_cpd_2 = TensorCPD(fmat=[np.arange(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2), mode_names=mode_names_2)
        assert t_cpd_1 == t_cpd_2

        # --------------------- Not equal because of Kruskal Rank
        ft_shape_2 = ft_shape_1
        R_2 = R_1 + 1
        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.arange(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        assert t_cpd_1 != t_cpd_2

        # --------------------- Not equal because of ft_shape
        ft_shape_2 = tuple(i+1 for i in ft_shape_1)
        R_2 = R_1
        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.arange(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        assert t_cpd_1 != t_cpd_2

        # --------------------- Not equal because of core values
        ft_shape_2 = ft_shape_1
        R_2 = R_1
        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.arange(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.zeros(R_2))
        assert t_cpd_1 != t_cpd_2

        # --------------------- Not equal because of fmat values
        ft_shape_2 = ft_shape_1
        R_2 = R_1
        t_cpd_1 = TensorCPD(fmat=[np.arange(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.ones(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        assert t_cpd_1 != t_cpd_2

        # --------------------- Not equal because of mode names
        ft_shape_2 = ft_shape_1
        R_2 = R_1
        mode_names_2 = ["modified-{}".format(name) for name in mode_names_1]
        t_cpd_1 = TensorCPD(fmat=[np.ones(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1), mode_names=mode_names_1)
        t_cpd_2 = TensorCPD(fmat=[np.ones(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2), mode_names=mode_names_2)
        assert t_cpd_1 != t_cpd_2
        t_cpd_2.set_mode_names(mode_names=new_mode_names)
        assert t_cpd_1 == t_cpd_2

        # --------------------- Not equal to the because of mode index
        ft_shape_2 = ft_shape_1
        R_2 = R_1
        t_cpd_1 = TensorCPD(fmat=[np.ones(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.ones(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        t_cpd_1.set_mode_index(mode_index=new_mode_index)
        assert t_cpd_1 != t_cpd_2

        # --------------------- Not equal to the because it is an instance of another class
        ft_shape_2 = ft_shape_1
        R_2 = R_1
        t_cpd_1 = TensorCPD(fmat=[np.ones(i * R_1).reshape(i, R_1) for i in ft_shape_1], core_values=np.ones(R_1))
        t_cpd_2 = TensorCPD(fmat=[np.ones(i * R_2).reshape(i, R_2) for i in ft_shape_2], core_values=np.ones(R_2))
        tensor_full = t_cpd_1.reconstruct()
        assert t_cpd_1 != tensor_full
        tensor_full = t_cpd_2.reconstruct()
        assert t_cpd_1 != tensor_full

    def test_addition(self):
        """ Test for summation of tensors in kruskal form """
        ft_shape = (4, 5, 6)
        R_1 = 2
        R_2 = 3
        mode_names = ["frequency", "time", "channel"]
        new_mode_index = {i: ["index" for _ in range(ft_shape[i])] for i in range(len(ft_shape))}
        core_values_1 = np.ones(R_1)
        core_values_2 = np.ones(R_2) * 2
        core_values_res = np.concatenate((core_values_1, core_values_2))
        fmat_1 = [np.arange(i * R_1).reshape(i, R_1) for i in ft_shape]
        fmat_2 = [np.ones(i * R_2).reshape(i, R_2) for i in ft_shape]
        fmat_res = [np.concatenate((fmat_1[i], fmat_2[i]), axis=1) for i in range(len(ft_shape))]

        t_cpd_1 = TensorCPD(fmat=fmat_1, core_values=core_values_1)
        t_cpd_2 = TensorCPD(fmat=fmat_2, core_values=core_values_2)
        t_cpd_res = TensorCPD(fmat=fmat_res, core_values=core_values_res)
        t_cpd_sum = t_cpd_1 + t_cpd_2
        assert t_cpd_sum == t_cpd_res

        t_cpd_1 = TensorCPD(fmat=fmat_1, core_values=core_values_1, mode_names=mode_names)
        t_cpd_2 = TensorCPD(fmat=fmat_2, core_values=core_values_2)
        t_cpd_res = TensorCPD(fmat=fmat_res, core_values=core_values_res)
        t_cpd_sum = t_cpd_1 + t_cpd_2
        assert t_cpd_sum == t_cpd_res

        #---- Tests that should fail
        with pytest.raises(TypeError):
            # wrong data type
            assert TensorCPD(fmat=fmat_1, core_values=core_values_1) + np.ones(R_2)

        with pytest.raises(TensorTopologyError):
            # wrong topology which is determined by the ft_shape or number of dimensions
            ft_shape_new = ft_shape + (6,)
            fmat_2_new = [np.ones(i * R_2).reshape(i, R_2) for i in ft_shape_new]
            t_cpd_1 = TensorCPD(fmat=fmat_1, core_values=core_values_1)
            t_cpd_2 = TensorCPD(fmat=fmat_2_new, core_values=core_values_2)
            assert t_cpd_1 + t_cpd_2

        with pytest.raises(ModeError):
            # wrong mode index
            t_cpd_1 = TensorCPD(fmat=fmat_1, core_values=core_values_1)
            t_cpd_2 = TensorCPD(fmat=fmat_2, core_values=core_values_2).set_mode_index(mode_index=new_mode_index)
            assert t_cpd_1 + t_cpd_2

    def test_repr(self):
        ft_shape = (3, 4, 5)
        R = 2
        core_values = np.ones(R)
        fmat = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        tensor_cpd = TensorCPD(core_values=core_values, fmat=fmat)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        print(repr(tensor_cpd))
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_copy(self):
        """ Tests for creation a copy of TensorCPD object """
        ft_shape = (3, 4, 5)  # define shape of the tensor in full form
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        tensor_cpd_copy = tensor_cpd.copy()

        # tests that the values are the same but not a reference
        assert tensor_cpd_copy is not tensor_cpd

        np.testing.assert_array_equal(tensor_cpd_copy._core_values, core_values)
        np.testing.assert_array_equal(tensor_cpd_copy._core_values, tensor_cpd._core_values)
        assert tensor_cpd_copy._core_values is not tensor_cpd._core_values

        assert tensor_cpd_copy.core is not tensor_cpd.core
        assert isinstance(tensor_cpd_copy.core, Tensor)
        np.testing.assert_array_equal(tensor_cpd_copy.core.data, tensor_cpd.core.data)
        assert tensor_cpd_copy.core.data is not tensor_cpd.core.data

        for mode in range(tensor_cpd.order):
            np.testing.assert_array_equal(tensor_cpd_copy.fmat[mode], fmat_list[mode])
            np.testing.assert_array_equal(tensor_cpd_copy.fmat[mode], tensor_cpd.fmat[mode])
            assert tensor_cpd_copy.fmat[mode] is not tensor_cpd.fmat[mode]

    def test_reconstruct(self):
        """ Tests for reconstruction TensorCPD object into the full form (Tensor) """
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
        true_data = np.array([[[225., 555., 885., 1215.],
                               [555., 1425., 2295., 3165.],
                               [885., 2295., 3705., 5115.]],

                              [[555., 1425., 2295., 3165.],
                               [1425., 4131., 6837., 9543.],
                               [2295., 6837., 11379., 15921.]]]
                             )
        ft_shape = true_data.shape  # define shape of the tensor in full form
        R = 6                       # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)

        # ------ basic tests on getting correct results after reconstruction
        tensor_rec = tensor_cpd.reconstruct()
        assert isinstance(tensor_rec, Tensor)
        np.testing.assert_array_equal(tensor_rec.data, true_data)
        assert (tensor_rec.ft_shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for consecutive reconstructions: results should be the same but different objects
        tensor_rec_1 = tensor_cpd.reconstruct()
        tensor_rec_2 = tensor_cpd.reconstruct()
        np.testing.assert_array_equal(tensor_rec_1.data, true_data)
        np.testing.assert_array_equal(tensor_rec_1.data, tensor_rec_2.data)
        assert tensor_rec_1 is not tensor_rec_2

        # ------ tests for chaining methods
        new_mode_names = {0: 'frequency',
                          1: 'time',
                          2: 'channel'
                          }
        mode = 0
        new_dim_size = 7
        matrix = np.arange(new_dim_size * ft_shape[mode]).reshape(new_dim_size, ft_shape[mode])

        tensor_rec = tensor_cpd.reconstruct().set_mode_names(mode_names=new_mode_names)
        for i, mode_name in enumerate(tensor_rec.mode_names):
            assert (mode_name == new_mode_names[i])

        new_name = 'age'
        tensor_rec = tensor_cpd.reconstruct().mode_n_product(matrix, mode=mode, new_name=new_name)
        new_shape = [i for i in ft_shape]
        new_shape[mode] = new_dim_size
        new_shape = tuple(new_shape)
        new_mode_names = true_default_mode_names
        new_mode_names[mode] = new_name
        assert (tensor_rec.shape == new_shape)
        assert (tensor_rec.mode_names == new_mode_names)

    def test_reconstruct_with_meta(self):
        """ Tests for keeping meta data about modes """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        mode_names = ["country", "year", "month"]
        mode_index ={0: ['UK', 'RUS'],
                     1: [2005, 2015, 2010],
                     2: ['Jan', 'Feb', 'Mar', 'Apr']}
        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values, mode_names=mode_names)
        tensor_cpd.set_mode_index(mode_index=mode_index)

        tensor = tensor_cpd.reconstruct(keep_meta=2)
        assert tensor.modes == tensor_cpd.modes

        tensor = tensor_cpd.reconstruct(keep_meta=1)
        assert all([tensor.modes[i].name == tensor_cpd.modes[i].name for i in range(tensor.order)])
        assert all([tensor.modes[i].index is None for i in range(tensor.order)])

        tensor = tensor_cpd.reconstruct(keep_meta=0)
        tensor_cpd.reset_mode_name()
        tensor_cpd.reset_mode_index()
        assert tensor.modes == tensor_cpd.modes

    def test_set_mode_names(self):
        """ Tests for `set_mode_names` method """
        ft_shape = (3, 4, 5)  # define shape of the tensor in full form
        true_order = len(ft_shape)
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        init_names = ["country", "year", "month"]
        mode_names = {i: name for i, name in enumerate(init_names)}

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        tensor_cpd.set_mode_names(mode_names)
        tensor_cpd_true = TensorCPD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        assert all([tensor_cpd.modes[i].name == tensor_cpd_true.modes[i].name for i in range(tensor_cpd.order)])

        # ------ tests that should FAIL for new mode names being incorrectly defined for renaming
        with pytest.raises(ModeError):
            # too many mode names
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order + 1)}
            tensor_cpd.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # incorrect type of keys (not integers)
            incorrect_new_mode_names = {"{}-mode".format(mode): mode for mode in range(true_order)}
            tensor_cpd.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(true_order - 2, true_order + 1)}
            tensor_cpd.set_mode_names(mode_names=incorrect_new_mode_names)

        with pytest.raises(ModeError):
            # key value is set to be negative
            incorrect_new_mode_names = {mode: "{}-mode".format(mode) for mode in range(-1, true_order - 1)}
            tensor_cpd.set_mode_names(mode_names=incorrect_new_mode_names)

    def test_reset_mode_name(self):
        """ Tests for `reset_mode_name` method """
        ft_shape = (3, 4, 5)  # define shape of the tensor in full form
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        init_names = ["country", "year", "month"]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        tensor_cpd.reset_mode_name()
        tensor_cpd_true = TensorCPD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_cpd.modes[i].name == tensor_cpd_true.modes[i].name for i in range(tensor_cpd.order)])

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        tensor_cpd.reset_mode_name(mode=0)
        init_names = ["mode-0", "year", "month"]
        tensor_cpd_true = TensorCPD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        assert all([tensor_cpd.modes[i].name == tensor_cpd_true.modes[i].name for i in range(tensor_cpd.order)])

    def test_set_mode_index(self):
        """ Tests for `set_mode_index` method """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        true_order = len(ft_shape)
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_cpd.set_mode_index(mode_index=mode_index)
        assert all([tensor_cpd.modes[i].index == mode_index[i] for i in range(tensor_cpd.order)])

        # ------ tests that should FAIL for new mode index being incorrectly defined for renaming
        with pytest.raises(ModeError):
            # too many lists of indices provided
            mode_index = {i: ["index"] for i in range(len(ft_shape) + 1)}
            tensor_cpd.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # incorrect type of keys (not integers)
            mode_index = {"index".format(mode): mode for mode in range(true_order)}
            tensor_cpd.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            wrong_key = true_order + 1
            mode_index = {wrong_key: ["idx"]}
            tensor_cpd.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # key value exceeds the order of a tensor
            wrong_key = -1
            mode_index = {wrong_key: ["idx"]}
            tensor_cpd.set_mode_index(mode_index=mode_index)

        with pytest.raises(ModeError):
            # not enough indices for the length of the mode
            mode_index = {0: ["idx"]}
            tensor_cpd.set_mode_index(mode_index=mode_index)

    def test_reset_mode_index(self):
        """ Tests for `reset_mode_index` method """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        R = 2  # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_cpd.set_mode_index(mode_index=mode_index)
        tensor_cpd.reset_mode_index()
        tensor_cpd_2 = TensorCPD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_cpd.modes[i].index == tensor_cpd_2.modes[i].index for i in range(tensor_cpd.order)])

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"]}
        tensor_cpd.set_mode_index(mode_index=mode_index)
        tensor_cpd.reset_mode_index(mode=0)
        tensor_cpd_2 = TensorCPD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_cpd.modes[i].index == tensor_cpd_2.modes[i].index for i in range(tensor_cpd.order)])


class TestTensorTKD:
    """ Tests for the TensorTKD class """

    def test_init(self):
        """ Tests for the TensorTKD constructor """
        ft_shape = (5, 6, 7)    # define shape of the tensor in full form
        ml_rank = (2, 3, 4)     # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        true_orig_fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode in range(len(ft_shape))]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values)
        assert isinstance(tensor_tkd.fmat, list)
        assert tensor_tkd.order == len(fmat_list)
        assert isinstance(tensor_tkd.rank, tuple)
        assert (tensor_tkd.rank == ml_rank)
        assert isinstance(tensor_tkd._core_values, np.ndarray)
        np.testing.assert_array_equal(tensor_tkd._core_values, core_values)
        assert tensor_tkd._core_values is not core_values

        # ------ tests for factor matrices
        for mode, fmat in enumerate(tensor_tkd.fmat):
            # check that values are the same but there are not references
            np.testing.assert_array_equal(fmat, fmat_list[mode])
            assert fmat is not fmat_list[mode]

            # check that changes to the matrices  have no affect on the TensorCPD
            # (double check for not being references)
            fmat_list[mode] = fmat_list[mode] * 2
            np.testing.assert_array_equal(fmat, true_orig_fmat_list[mode])
            assert fmat is not true_orig_fmat_list[mode]

        # ------ tests for core
        assert isinstance(tensor_tkd.core, Tensor)
        np.testing.assert_array_equal(tensor_tkd.core.data, core_values)

    def test_init_fail(self):
        """ Tests for incorrect input data for the TensorTKD constructor """
        # ------ the following tests should FAIL
        ft_shape = (5, 6, 7)  # define shape of the tensor in full form
        ml_rank = (2, 3, 4)  # define multi-linear rank of a tensor in Tucker form
        correct_core_values = np.ones(ml_rank)
        correct_fmat = [np.ones([ft_shape[mode], ml_rank[mode]]) for mode in range(len(ft_shape))]

        # core_values should be in form of numpy array
        with pytest.raises(TypeError):
            incorrect_core_values = list(correct_core_values)
            TensorTKD(fmat=correct_fmat, core_values=incorrect_core_values)

        # factor matrices should be passed as a list
        with pytest.raises(TypeError):
            incorrect_fmat = tuple(correct_fmat)
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

        # all factor matrices should be in form of numpy array
        with pytest.raises(TypeError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            mode = 0
            incorrect_fmat[mode] = list(incorrect_fmat[mode])
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

        # all factor matrices should be a 2-dimensional numpy array
        with pytest.raises(TensorTopologyError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            mode = 0
            incorrect_fmat[mode] = np.ones([ft_shape[mode], ml_rank[mode], 2])
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

        # Not enough factor matrices for the specified core tensor
        with pytest.raises(TensorTopologyError):
            incorrect_core_values = np.ones(correct_core_values.shape + (2,))
            TensorTKD(fmat=correct_fmat, core_values=incorrect_core_values)

        # number of columns of some factor matrices does not match the size of the corresponding mode of the core
        with pytest.raises(TensorTopologyError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = incorrect_fmat[0].T
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

    def test_equal(self):
        """ Test for tensors in tucker form being equal """
        create_fmat = lambda func, shape, R: [func(shape[i] * R[i]).reshape(shape[i], R[i]) for i in range(len(shape))]

        ft_shape = (5, 6, 7)
        ft_shape_new = tuple(i + 1 for i in ft_shape)
        ml_rank = (2, 3, 4)
        ml_rank_new = tuple(i + 1 for i in ml_rank)
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_size_new = reduce(lambda x, y: x * y, ml_rank_new)

        mode_names_1 = ["frequency", "time", "channel"]
        mode_names_2 = [name for name in mode_names_1]
        new_mode_names = {i: "{}".format(mode_names_1[i]) for i in range(len(ft_shape))}
        new_mode_index = {i: ["index" for _ in range(ft_shape[i])] for i in range(len(ft_shape))}

        core_values_1 = np.arange(core_size).reshape(ml_rank)
        fmat_1 = create_fmat(np.arange, ft_shape, ml_rank)

        core_values_2 = core_values_1.copy()
        fmat_2 = [fmat.copy() for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        assert tensor_tkd_1 == tensor_tkd_2

        tensor_tkd_1.set_mode_names(mode_names=new_mode_names)
        tensor_tkd_2.set_mode_names(mode_names=new_mode_names)
        assert tensor_tkd_1 == tensor_tkd_2

        tensor_tkd_1.set_mode_index(mode_index=new_mode_index)
        tensor_tkd_2.set_mode_index(mode_index=new_mode_index)
        assert tensor_tkd_1 == tensor_tkd_2

        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1, mode_names=mode_names_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2, mode_names=mode_names_2)
        assert tensor_tkd_1 == tensor_tkd_2

        # --------------------- Not equal because of multi-linear Rank
        core_values_2 = np.arange(core_size_new).reshape(ml_rank_new)
        fmat_2 = create_fmat(np.arange, ft_shape, ml_rank_new)
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        assert tensor_tkd_1 != tensor_tkd_2

        # --------------------- Not equal because of ft_shape
        core_values_2 = core_values_1.copy()
        fmat_2 = create_fmat(np.arange, ft_shape_new, ml_rank)
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        assert tensor_tkd_1 != tensor_tkd_2

        # --------------------- Not equal because of core values
        core_values_2 = core_values_1.copy() * 2
        fmat_2 = [fmat.copy() for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        assert tensor_tkd_1 != tensor_tkd_2

        # --------------------- Not equal because of fmat values
        core_values_2 = core_values_1.copy()
        fmat_2 = [fmat.copy() * 2 for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        assert tensor_tkd_1 != tensor_tkd_2

        # --------------------- Not equal because of mode names
        mode_names_2 = ["modified-{}".format(name) for name in mode_names_1]
        core_values_2 = core_values_1.copy()
        fmat_2 = [fmat.copy() for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1, mode_names=mode_names_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2, mode_names=mode_names_2)
        assert tensor_tkd_1 != tensor_tkd_2
        tensor_tkd_2.set_mode_names(mode_names=new_mode_names)
        assert tensor_tkd_1 == tensor_tkd_2

        # --------------------- Not equal to the because of mode index
        core_values_2 = core_values_1.copy()
        fmat_2 = [fmat.copy() for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        tensor_tkd_1.set_mode_index(mode_index=new_mode_index)
        assert tensor_tkd_1 != tensor_tkd_2

        # --------------------- Not equal to the because it is an instance of another class
        core_values_2 = core_values_1.copy()
        fmat_2 = [fmat.copy() for fmat in fmat_1]
        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        tensor_full = tensor_tkd_2.reconstruct()
        assert tensor_tkd_1 != tensor_full

    def test_addition(self):
        """ Test for summation of tensors in tucker form """
        create_fmat = lambda func, shape, R: [func(shape[i] * R[i]).reshape(shape[i], R[i]) for i in range(len(shape))]

        ft_shape = (4, 5, 6)
        ml_rank_1 = (2, 2, 2)
        ml_rank_2 = (3, 3, 3)
        core_size_1 = reduce(lambda x, y: x * y, ml_rank_1)

        mode_names = ["frequency", "time", "channel"]
        new_mode_index = {i: ["index" for _ in range(ft_shape[i])] for i in range(len(ft_shape))}

        core_values_1 = np.arange(core_size_1).reshape(ml_rank_1)
        core_values_2 = np.ones(ml_rank_2) * 2
        core_values_res = np.array([[[0., 1., 0., 0., 0.],
                                     [2., 3., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.]],

                                    [[4., 5., 0., 0., 0.],
                                     [6., 7., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.]],

                                    [[0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.]],

                                    [[0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.]],

                                    [[0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.],
                                     [0., 0., 2., 2., 2.]]])

        fmat_1 = create_fmat(np.arange, ft_shape, ml_rank_1)
        fmat_2 = create_fmat(np.ones, ft_shape, ml_rank_2)
        fmat_res = [np.concatenate((fmat_1[i], fmat_2[i]), axis=1) for i in range(len(ft_shape))]

        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        tensor_tkd_res = TensorTKD(fmat=fmat_res, core_values=core_values_res)
        tensor_tkd_sum = tensor_tkd_1 + tensor_tkd_2
        assert tensor_tkd_sum == tensor_tkd_res

        tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1, mode_names=mode_names)
        tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2)
        tensor_tkd_res = TensorTKD(fmat=fmat_res, core_values=core_values_res)
        tensor_tkd_sum = tensor_tkd_1 + tensor_tkd_2
        assert tensor_tkd_sum == tensor_tkd_res

        # ---- Tests that should fail
        with pytest.raises(TypeError):
            # wrong data type
            assert TensorTKD(fmat=fmat_1, core_values=core_values_1) + np.ones(ml_rank_1)

        with pytest.raises(TensorTopologyError):
            # wrong topology which is determined by the ft_shape or number of dimensions
            ft_shape_new = tuple(i + 1 for i in ft_shape)
            fmat_2_new = create_fmat(np.ones, ft_shape_new, ml_rank_2)
            tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
            tensor_tkd_2 = TensorTKD(fmat=fmat_2_new, core_values=core_values_2)
            assert tensor_tkd_1 + tensor_tkd_2

        with pytest.raises(ModeError):
            # wrong mode index
            tensor_tkd_1 = TensorTKD(fmat=fmat_1, core_values=core_values_1)
            tensor_tkd_2 = TensorTKD(fmat=fmat_2, core_values=core_values_2).set_mode_index(mode_index=new_mode_index)
            assert tensor_tkd_1 + tensor_tkd_2

    def test_repr(self):
        shape = (5, 6, 7)
        rank = (2, 3, 4)
        core_size = reduce(lambda x, y: x * y, rank)
        core_values = np.arange(core_size).reshape(rank)
        fmat = [np.arange(shape[mode] * rank[mode]).reshape(shape[mode], rank[mode]) for mode in range(len(shape))]
        tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        print(repr(tensor_tkd))
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_copy(self):
        """ Tests for creation a copy of TensorTKD object """
        ft_shape = (2, 3, 4)    # define shape of the tensor in full form
        ml_rank = (5, 6, 7)     # define multi-linear rank of a tensor in Tucker form
        core_values = np.ones(ml_rank)
        fmat = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode in range(len(ft_shape))]

        tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)
        tensor_tkd_copy = tensor_tkd.copy()

        # tests that the values are the same but not a reference
        assert tensor_tkd_copy is not tensor_tkd

        np.testing.assert_array_equal(tensor_tkd_copy._core_values, core_values)
        np.testing.assert_array_equal(tensor_tkd_copy._core_values, tensor_tkd._core_values)
        assert tensor_tkd_copy._core_values is not tensor_tkd._core_values

        assert tensor_tkd_copy.core is not tensor_tkd.core
        assert isinstance(tensor_tkd_copy.core, Tensor)
        np.testing.assert_array_equal(tensor_tkd_copy.core.data, tensor_tkd.core.data)
        assert tensor_tkd_copy.core.data is not tensor_tkd.core.data

        for mode in range(tensor_tkd.order):
            np.testing.assert_array_equal(tensor_tkd_copy.fmat[mode], fmat[mode])
            np.testing.assert_array_equal(tensor_tkd_copy.fmat[mode], tensor_tkd.fmat[mode])
            assert tensor_tkd_copy.fmat[mode] is not tensor_tkd.fmat[mode]

    def test_reconstruct(self):
        """ Tests for reconstruction TensorTKD object into the full form (Tensor) """
        ft_shape = (2, 3, 4)    # define shape of the tensor in full form
        ml_rank = (5, 6, 7)     # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        fmat = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                               in range(len(ft_shape))]
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
        true_data = np.array([[[491400,  1628200,  2765000,  3901800],
                               [1609020,  5330080,  9051140, 12772200],
                               [2726640,  9031960, 15337280, 21642600]],

                              [[1389150,  4596200,  7803250, 11010300],
                               [4507020, 14906780, 25306540, 35706300],
                               [7624890, 25217360, 42809830, 60402300]]]
                             )

        tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)

        # ------ basic tests on getting correct results after reconstruction
        tensor_rec = tensor_tkd.reconstruct()
        assert isinstance(tensor_rec, Tensor)
        np.testing.assert_array_equal(tensor_rec.data, true_data)
        assert (tensor_rec.ft_shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for consecutive reconstructions: results should be the same but different objects
        tensor_rec_1 = tensor_tkd.reconstruct()
        tensor_rec_2 = tensor_tkd.reconstruct()
        np.testing.assert_array_equal(tensor_rec_1.data, true_data)
        np.testing.assert_array_equal(tensor_rec_1.data, tensor_rec_2.data)
        assert tensor_rec_1 is not tensor_rec_2

        # ------ tests for chaining methods
        new_mode_names = {0: 'frequency',
                          1: 'time',
                          2: 'channel'
                          }

        mode = 0
        new_dim_size = 7
        matrix = np.arange(new_dim_size * ft_shape[mode]).reshape(new_dim_size, ft_shape[mode])

        tensor_rec = tensor_tkd.reconstruct().set_mode_names(mode_names=new_mode_names)
        for i, mode_name in enumerate(tensor_rec.mode_names):
            assert (mode_name == new_mode_names[i])

        new_name = 'age'
        tensor_rec = tensor_tkd.reconstruct().mode_n_product(matrix, mode=mode, new_name=new_name)
        new_shape = [i for i in ft_shape]
        new_shape[mode] = new_dim_size
        new_shape = tuple(new_shape)
        new_mode_names = true_default_mode_names
        new_mode_names[mode] = new_name
        assert (tensor_rec.shape == new_shape)
        assert (tensor_rec.mode_names == new_mode_names)

    def test_reconstruct_with_meta(self):
        """ Tests for keeping meta data about modes """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        ml_rank = (5, 6, 7)  # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                in range(len(ft_shape))]
        mode_names = ["country", "year", "month"]
        mode_index ={0: ['UK', 'RUS'],
                     1: [2005, 2015, 2010],
                     2: ['Jan', 'Feb', 'Mar', 'Apr']}
        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values, mode_names=mode_names)
        tensor_tkd.set_mode_index(mode_index=mode_index)

        tensor = tensor_tkd.reconstruct(keep_meta=2)
        assert tensor.modes == tensor_tkd.modes

        tensor = tensor_tkd.reconstruct(keep_meta=1)
        assert all([tensor.modes[i].name == tensor_tkd.modes[i].name for i in range(tensor.order)])
        assert all([tensor.modes[i].index is None for i in range(tensor.order)])

        tensor = tensor_tkd.reconstruct(keep_meta=0)
        tensor_tkd.reset_mode_name()
        tensor_tkd.reset_mode_index()
        assert tensor.modes == tensor_tkd.modes

    def test_set_mode_names(self):
        """ Tests for `set_mode_names` method """
        ft_shape = (5, 6, 7)  # define shape of the tensor in full form
        ml_rank = (2, 3, 4)  # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        true_orig_fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                               in range(len(ft_shape))]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]
        init_names = ["country", "year", "month"]
        mode_names = {i: name for i, name in enumerate(init_names)}

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values)
        tensor_tkd.set_mode_names(mode_names)
        tensor_tkd_true = TensorTKD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        assert all([tensor_tkd.modes[i].name == tensor_tkd_true.modes[i].name for i in range(tensor_tkd.order)])

    def test_reset_mode_name(self):
        """ Tests for `reset_mode_name` method """
        ft_shape = (5, 6, 7)  # define shape of the tensor in full form
        ml_rank = (2, 3, 4)  # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        true_orig_fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                               in range(len(ft_shape))]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]
        init_names = ["country", "year", "month"]

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        tensor_tkd.reset_mode_name()
        tensor_tkd_true = TensorTKD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_tkd.modes[i].name == tensor_tkd_true.modes[i].name for i in range(tensor_tkd.order)])

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        tensor_tkd.reset_mode_name(mode=0)
        init_names = ["mode-0", "year", "month"]
        tensor_tkd_true = TensorTKD(fmat=fmat_list, core_values=core_values, mode_names=init_names)
        assert all([tensor_tkd.modes[i].name == tensor_tkd_true.modes[i].name for i in range(tensor_tkd.order)])

    def test_set_mode_index(self):
        """ Tests for `set_mode_index` method """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        ml_rank = (2, 3, 4)  # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        true_orig_fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                               in range(len(ft_shape))]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_tkd.set_mode_index(mode_index=mode_index)
        assert all([tensor_tkd.modes[i].index == mode_index[i] for i in range(tensor_tkd.order)])

    def test_reset_mode_index(self):
        """ Tests for `reset_mode_index` method """
        ft_shape = (2, 3, 4)  # define shape of the tensor in full form
        ml_rank = (2, 3, 4)  # define multi-linear rank of a tensor in Tucker form
        core_size = reduce(lambda x, y: x * y, ml_rank)
        core_values = np.arange(core_size).reshape(ml_rank)
        true_orig_fmat_list = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode], ml_rank[mode]) for mode
                               in range(len(ft_shape))]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_tkd.set_mode_index(mode_index=mode_index)
        tensor_tkd.reset_mode_index()
        tensor_tkd_2 = TensorTKD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_tkd.modes[i].index == tensor_tkd_2.modes[i].index for i in range(tensor_tkd.order)])

        tensor_tkd = TensorTKD(fmat=fmat_list, core_values=core_values)
        mode_index = {0: ["idx1", "idx2"]}
        tensor_tkd.set_mode_index(mode_index=mode_index)
        tensor_tkd.reset_mode_index(mode=0)
        tensor_tkd_2 = TensorTKD(fmat=fmat_list, core_values=core_values)
        assert all([tensor_tkd.modes[i].index == tensor_tkd_2.modes[i].index for i in range(tensor_tkd.order)])


class TestTensorTT:
    """ Tests for the TensorTT class """

    def test_init(self):
        """ Tests for the TensorTT constructor """
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        true_orig_core_values = [core.copy() for core in core_values]
        true_tt_rank = (r1, r2)
        true_ft_shape = (I, J, K)
        true_order = len(true_ft_shape)
        true_default_mode_names_2d = ['mode-0', 'mode-1']
        true_default_mode_names_3d = ['mode-0', 'mode-1', 'mode-2']

        tensor_tt = TensorTT(core_values=core_values)
        # ------ tests for types of data being correct
        assert isinstance(tensor_tt._core_values, list)
        assert isinstance(tensor_tt.ft_shape, tuple)
        assert isinstance(tensor_tt.cores, list)
        assert isinstance(tensor_tt.rank, tuple)
        for i, core in enumerate(tensor_tt.cores):
            assert isinstance(core, Tensor)
            assert isinstance(tensor_tt.core(i), Tensor)
            assert isinstance(tensor_tt._core_values[i], np.ndarray)

        # ------ tests for data being correct
        assert (tensor_tt.rank == true_tt_rank)
        assert (tensor_tt.order == true_order)

        # check that values are the same but they are not a references
        assert (tensor_tt.ft_shape == true_ft_shape)
        assert tensor_tt.ft_shape is not true_ft_shape
        for i, core in enumerate(tensor_tt.cores):
            np.testing.assert_array_equal(core.data, core_values[i])
            np.testing.assert_array_equal(tensor_tt._core_values[i], core_values[i])
            assert core.data is not core_values[i]
            assert tensor_tt._core_values[i] is not core_values[i]
            assert (core.order == 2) or (core.order == 3)   # cores should be either matrices or 3d arrays
            if core.order == 2:
                assert core.mode_names == true_default_mode_names_2d
            if core.order == 3:
                assert core.mode_names == true_default_mode_names_3d

        # double check for not being references
        for i, core in enumerate(tensor_tt.cores):
            core_values[i] = core_values[i] * 2
            np.testing.assert_array_equal(core.data, true_orig_core_values[i])
            np.testing.assert_array_equal(tensor_tt._core_values[i], true_orig_core_values[i])

        with pytest.raises(IndexError):
            order = tensor_tt.order
            tensor_tt.core(i=order)

        with pytest.raises(IndexError):
            order = tensor_tt.order
            tensor_tt.core(i=(-order))

    def test_init_fail(self):
        """ Tests for incorrect input data for the TensorTT constructor """
        # ------ the following tests should FAIL
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        correct_core_1 = np.arange(I * r1).reshape(I, r1)
        correct_core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        correct_core_3 = np.arange(r2 * K).reshape(r2, K)
        correct_core_values = [correct_core_1, correct_core_2, correct_core_3]
        correct_ft_shape = (I, J, K)


        # core_values should be a list of numpy arrays
        with pytest.raises(TypeError):
            incorrect_core_values = np.arange(5)
            TensorTT(core_values=incorrect_core_values)

        # all elements in core_values should be numpy arrays
        with pytest.raises(TypeError):
            incorrect_core_values = [[1], [2], [3]]
            TensorTT(core_values=incorrect_core_values)

        # not enough elements in core_values for the specified ft_shape
        with pytest.raises(TensorTopologyError):
            incorrect_core_values = [correct_core_1, correct_core_2]
            TensorTT(core_values=incorrect_core_values)

        # first and last element of core_values should be 2-dimensional arrays
        with pytest.raises(TensorTopologyError):
            shape = (2, 2, 2)
            incorrect_core_values = [np.ones(shape) for _ in range(len(correct_ft_shape))]
            TensorTT(core_values=incorrect_core_values)

        # All but first and last element of core_values should be 3-dimensional arrays
        with pytest.raises(TensorTopologyError):
            shape = (2, 2)
            incorrect_core_values = [np.ones(shape) for _ in range(len(correct_ft_shape))]
            TensorTT(core_values=incorrect_core_values)

        # Last dimension of core_values[i] should be the same as the first dimension of core_values[i+1]
        with pytest.raises(TensorTopologyError):
            incorrect_core_values = [np.ones((2, 3)), np.ones((3, 4, 5)), np.ones((6, 8))]
            TensorTT(core_values=incorrect_core_values)

    def test_equal(self):
        """ Test for tensors in tensor train form being equal """
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        ft_shape = (I, J, K)
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        mode_names_1 = ["frequency", "time", "channel"]
        mode_names_2 = [name for name in mode_names_1]
        new_mode_names = {i: "{}".format(mode_names_1[i]) for i in range(len(ft_shape))}
        new_mode_index = {i: ["index" for _ in range(ft_shape[i])] for i in range(len(ft_shape))}

        core_values_1 = [core_1, core_2, core_3]
        core_values_2 = [core.copy() for core in core_values_1]

        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        assert tensor_tt_1 == tensor_tt_2

        tensor_tt_1.set_mode_names(mode_names=new_mode_names)
        tensor_tt_2.set_mode_names(mode_names=new_mode_names)
        assert tensor_tt_1 == tensor_tt_2

        tensor_tt_1.set_mode_index(mode_index=new_mode_index)
        tensor_tt_2.set_mode_index(mode_index=new_mode_index)
        assert tensor_tt_1 == tensor_tt_2

        tensor_tt_1 = TensorTT(core_values=core_values_1, mode_names=mode_names_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2, mode_names=mode_names_2)
        assert tensor_tt_1 == tensor_tt_2

        # --------------------- Not equal because of core values
        core_values_2 = [core.copy() * 2 for core in core_values_1]
        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        assert tensor_tt_1 != tensor_tt_2

        # --------------------- Not equal because of mode names
        mode_names_2 = ["modified-{}".format(name) for name in mode_names_1]
        core_values_2 = [core.copy() for core in core_values_1]
        tensor_tt_1 = TensorTT(core_values=core_values_1, mode_names=mode_names_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2, mode_names=mode_names_2)
        assert tensor_tt_1 != tensor_tt_2
        tensor_tt_2.set_mode_names(mode_names=new_mode_names)
        assert tensor_tt_1 == tensor_tt_2

        # --------------------- Not equal to the because of mode index
        core_values_2 = [core.copy() for core in core_values_1]
        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        tensor_tt_2.set_mode_index(mode_index=new_mode_index)
        assert tensor_tt_1 != tensor_tt_2

        # --------------------- Not equal because of TT-Rank
        core_values_2 = [core.copy() for core in core_values_1]
        r1, J, r2 = core_values_2[1].shape
        core_values_2[0] = np.arange(I * (r1+1)).reshape(I, (r1+1))
        core_values_2[1] = np.arange((r1+1) * J * r2).reshape((r1+1), J, r2)
        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        assert tensor_tt_1 != tensor_tt_2

        # --------------------- Not equal because of ft_shape
        core_values_2 = [core.copy() for core in core_values_1]
        r1, J, r2 = core_values_2[1].shape
        core_values_2[1] = np.arange(r1 * (J+1) * r2).reshape(r1, (J+1), r2)
        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        assert tensor_tt_1 != tensor_tt_2

        # --------------------- Not equal to the because it is an instance of another class
        core_values_2 = [core.copy() for core in core_values_1]
        tensor_tt_1 = TensorTT(core_values=core_values_1)
        tensor_tt_2 = TensorTT(core_values=core_values_2)
        tensor_full = tensor_tt_2.reconstruct()
        assert tensor_tt_1 != tensor_full

    def test_repr(self):
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        tensor_tt = TensorTT(core_values=core_values)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        print(repr(tensor_tt))
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_copy(self):
        """ Tests for creation a copy of TensorTT object """
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)
        tensor_tt = TensorTT(core_values=core_values)

        tensor_tt_copy = tensor_tt.copy()

        # tests that the values are the same but not a reference
        assert tensor_tt_copy is not tensor_tt
        assert tensor_tt_copy.ft_shape is not tensor_tt.ft_shape
        assert tensor_tt_copy.ft_shape == tensor_tt.ft_shape
        assert tensor_tt_copy.rank == tensor_tt.rank
        assert tensor_tt_copy.order == tensor_tt.order

        assert tensor_tt_copy._core_values is not tensor_tt._core_values
        for i in range(tensor_tt_copy.order):
            assert tensor_tt_copy._core_values[i] is not tensor_tt._core_values[i]
            np.testing.assert_array_equal(tensor_tt_copy._core_values[i], core_values[i])
            np.testing.assert_array_equal(tensor_tt_copy._core_values[i], tensor_tt._core_values[i])
            assert tensor_tt_copy.core(i) is not tensor_tt.core(i)
            np.testing.assert_array_equal(tensor_tt_copy.core(i).data, tensor_tt.core(i).data)

        assert tensor_tt_copy.cores is not tensor_tt.cores

    def test_reconstruct(self):
        """ Tests for reconstruction TensorTT object into the full form (Tensor) """
        r1, r2 = 2, 3
        I, J, K = 4, 5, 6
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)
        true_data = np.array([[[ 300,  348,  396,  444,  492,  540],
                               [ 354,  411,  468,  525,  582,  639],
                               [ 408,  474,  540,  606,  672,  738],
                               [ 462,  537,  612,  687,  762,  837],
                               [ 516,  600,  684,  768,  852,  936]],

                              [[ 960, 1110, 1260, 1410, 1560, 1710],
                               [1230, 1425, 1620, 1815, 2010, 2205],
                               [1500, 1740, 1980, 2220, 2460, 2700],
                               [1770, 2055, 2340, 2625, 2910, 3195],
                               [2040, 2370, 2700, 3030, 3360, 3690]],

                              [[1620, 1872, 2124, 2376, 2628, 2880],
                               [2106, 2439, 2772, 3105, 3438, 3771],
                               [2592, 3006, 3420, 3834, 4248, 4662],
                               [3078, 3573, 4068, 4563, 5058, 5553],
                               [3564, 4140, 4716, 5292, 5868, 6444]],

                              [[2280, 2634, 2988, 3342, 3696, 4050],
                               [2982, 3453, 3924, 4395, 4866, 5337],
                               [3684, 4272, 4860, 5448, 6036, 6624],
                               [4386, 5091, 5796, 6501, 7206, 7911],
                               [5088, 5910, 6732, 7554, 8376, 9198]]])
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
        tensor_tt = TensorTT(core_values=core_values)

        # ------ basic tests on getting correct results after reconstruction
        tensor_rec = tensor_tt.reconstruct()
        assert isinstance(tensor_rec, Tensor)
        np.testing.assert_array_equal(tensor_rec.data, true_data)
        assert (tensor_rec.ft_shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for consecutive reconstructions: results should be the same but different objects
        tensor_rec_1 = tensor_tt.reconstruct()
        tensor_rec_2 = tensor_tt.reconstruct()
        np.testing.assert_array_equal(tensor_rec_1.data, true_data)
        np.testing.assert_array_equal(tensor_rec_1.data, tensor_rec_2.data)
        assert tensor_rec_1 is not tensor_rec_2

        # ------ tests for chaining methods
        new_mode_names = {0: 'frequency',
                          1: 'time',
                          2: 'channel'
                          }

        mode = 0
        new_dim_size = 7
        matrix = np.arange(new_dim_size * ft_shape[mode]).reshape(new_dim_size, ft_shape[mode])

        tensor_rec = tensor_tt.reconstruct().set_mode_names(mode_names=new_mode_names)
        for i, mode_name in enumerate(tensor_rec.mode_names):
            assert (mode_name == new_mode_names[i])

        new_name = 'age'
        tensor_rec = tensor_tt.reconstruct().mode_n_product(matrix, mode=mode, new_name=new_name)
        new_shape = [i for i in ft_shape]
        new_shape[mode] = new_dim_size
        new_shape = tuple(new_shape)
        new_mode_names = true_default_mode_names
        new_mode_names[mode] = new_name
        assert (tensor_rec.shape == new_shape)
        assert (tensor_rec.mode_names == new_mode_names)

        # ------ tests for the 4th order Tensor
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2', 'mode-3']

        r1, r2, r3 = 2, 3, 4
        I, J, K, L = 5, 6, 7, 8
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K * r3).reshape(r2, K, r3)
        core_4 = np.arange(r3 * L).reshape(r3, L)
        core_values = [core_1, core_2, core_3, core_4]
        ft_shape = (I, J, K, L)
        tensor_tt = TensorTT(core_values=core_values)
        tensor_rec = tensor_tt.reconstruct()
        assert (tensor_rec.shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for the 5th order Tensor
        true_default_mode_names = ['mode-0', 'mode-1', 'mode-2', 'mode-3', 'mode-4']

        r1, r2, r3, r4 = 2, 3, 4, 5
        I, J, K, L, M = 5, 6, 7, 8, 9
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K * r3).reshape(r2, K, r3)
        core_4 = np.arange(r3 * L * r4).reshape(r3, L, r4)
        core_5 = np.arange(r4 * M).reshape(r4, M)
        core_values = [core_1, core_2, core_3, core_4, core_5]
        ft_shape = (I, J, K, L, M)
        tensor_tt = TensorTT(core_values=core_values)
        tensor_rec = tensor_tt.reconstruct()
        assert (tensor_rec.shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

    def test_reconstruct_with_meta(self):
        """ Tests for keeping meta data about modes """
        r1, r2 = 2, 3
        I, J, K = 2, 3, 4
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)
        mode_names = ["country", "year", "month"]
        mode_index ={0: ['UK', 'RUS'],
                     1: [2005, 2015, 2010],
                     2: ['Jan', 'Feb', 'Mar', 'Apr']}
        tensor_tt = TensorTT(core_values=core_values, mode_names=mode_names)
        tensor_tt.set_mode_index(mode_index=mode_index)

        tensor = tensor_tt.reconstruct(keep_meta=2)
        assert tensor.modes == tensor_tt.modes

        tensor = tensor_tt.reconstruct(keep_meta=1)
        assert all([tensor.modes[i].name == tensor_tt.modes[i].name for i in range(tensor.order)])
        assert all([tensor.modes[i].index is None for i in range(tensor.order)])

        tensor = tensor_tt.reconstruct(keep_meta=0)
        tensor_tt.reset_mode_name()
        tensor_tt.reset_mode_index()
        assert tensor.modes == tensor_tt.modes

    def test_set_mode_names(self):
        """ Tests for `set_mode_names` method """
        r1, r2 = 2, 3
        I, J, K = 2, 3, 4
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)
        init_names = ["country", "year", "month"]
        mode_names = {i: name for i, name in enumerate(init_names)}

        tensor_tkd = TensorTT(core_values=core_values)
        tensor_tkd.set_mode_names(mode_names)
        tensor_tkd_true = TensorTT(core_values=core_values, mode_names=init_names)
        assert all([tensor_tkd.modes[i].name == tensor_tkd_true.modes[i].name for i in range(tensor_tkd.order)])

    def test_reset_mode_name(self):
        """ Tests for `reset_mode_name` method """
        r1, r2 = 2, 3
        I, J, K = 2, 3, 4
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)
        init_names = ["country", "year", "month"]

        tensor_tt = TensorTT(core_values=core_values, mode_names=init_names)
        tensor_tt.reset_mode_name()
        tensor_tkd_true = TensorTT(core_values=core_values)
        assert all([tensor_tt.modes[i].name == tensor_tkd_true.modes[i].name for i in range(tensor_tt.order)])

        tensor_tt = TensorTT(core_values=core_values, mode_names=init_names)
        tensor_tt.reset_mode_name(mode=0)
        init_names = ["mode-0", "year", "month"]
        tensor_tt_true = TensorTT(core_values=core_values, mode_names=init_names)
        assert all([tensor_tt.modes[i].name == tensor_tt_true.modes[i].name for i in range(tensor_tt.order)])

    def test_set_mode_index(self):
        """ Tests for `set_mode_index` method """
        r1, r2 = 2, 3
        I, J, K = 2, 3, 4
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)

        tensor_tt = TensorTT(core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_tt.set_mode_index(mode_index=mode_index)
        assert all([tensor_tt.modes[i].index == mode_index[i] for i in range(tensor_tt.order)])

    def test_reset_mode_index(self):
        """ Tests for `reset_mode_index` method """
        r1, r2 = 2, 3
        I, J, K = 2, 3, 4
        core_1 = np.arange(I * r1).reshape(I, r1)
        core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
        core_3 = np.arange(r2 * K).reshape(r2, K)
        core_values = [core_1, core_2, core_3]
        ft_shape = (I, J, K)

        tensor_tt = TensorTT(core_values=core_values)
        mode_index = {0: ["idx1", "idx2"],
                      1: ["idx1", "idx2", "idx3"],
                      2: ["idx1", "idx2", "idx3", "idx4"]}
        tensor_tt.set_mode_index(mode_index=mode_index)
        tensor_tt.reset_mode_index()
        tensor_tt_2 = TensorTT(core_values=core_values)
        assert all([tensor_tt.modes[i].index == tensor_tt_2.modes[i].index for i in range(tensor_tt.order)])

        tensor_tt = TensorTT(core_values=core_values)
        mode_index = {0: ["idx1", "idx2"]}
        tensor_tt.set_mode_index(mode_index=mode_index)
        tensor_tt.reset_mode_index(mode=0)
        tensor_tt_2 = TensorTT(core_values=core_values)
        assert all([tensor_tt.modes[i].index == tensor_tt_2.modes[i].index for i in range(tensor_tt.order)])


def test_super_diag_tensor():
    """ Tests for creating super-diagonal tensor"""
    order = 3
    rank = 2
    correct_shape = (rank, ) * order
    true_default_data = np.array([[[1., 0.],
                                   [0., 0.]],

                                  [[0., 0.],
                                   [0., 1.]]])
    true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']
    correct_values = np.arange(rank)
    true_data = np.array([[[0., 0.],
                           [0., 0.]],

                          [[0., 0.],
                           [0., 1.]]])

    # ------ tests for default super diagonal tensor
    tensor = super_diag_tensor(correct_shape)
    assert isinstance(tensor, Tensor)
    np.testing.assert_array_equal(tensor.data, true_default_data)
    assert (tensor.mode_names == true_default_mode_names)

    # ------ tests for super diagonal tensor with custom values on the main diagonal
    tensor = super_diag_tensor(correct_shape, values=correct_values)
    assert isinstance(tensor, Tensor)
    np.testing.assert_array_equal(tensor.data, true_data)
    assert (tensor.mode_names == true_default_mode_names)

    # ------ tests that should Fail

    with pytest.raises(TypeError):
        # shape should be passed as tuple
        super_diag_tensor(shape=list(correct_shape))

    with pytest.raises(ValueError):
        # all values in shape should be the same
        incorrect_shape = [rank] * order
        incorrect_shape[1] = order+1
        super_diag_tensor(shape=tuple(incorrect_shape))

    with pytest.raises(ValueError):
        # values should be an one dimensional numpy array
        incorrect_values = np.ones([rank, rank])
        super_diag_tensor(shape=correct_shape, values=incorrect_values)

    with pytest.raises(ValueError):
        # too many values for the specified shape
        incorrect_values = np.ones(correct_shape[0]+1)
        super_diag_tensor(shape=correct_shape, values=incorrect_values)

    with pytest.raises(TypeError):
        # values should be a numpy array
        incorrect_values = [1] * correct_shape[0]
        super_diag_tensor(shape=correct_shape, values=incorrect_values)


def test_residual_tensor():
    """ Tests for computing/creating a residual tensor """
    true_default_mode_names = ['mode-0', 'mode-1', 'mode-2']

    # ------ tests for residual tensor with the Tensor
    array_3d = np.array([[[0,  1,  2,  3],
                          [4,  5,  6,  7],
                          [8,  9, 10, 11]],

                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]])
    true_residual_data = np.zeros(array_3d.shape)
    tensor_1 = Tensor(array=array_3d)
    tensor_2 = Tensor(array=array_3d)
    residual = residual_tensor(tensor_orig=tensor_1, tensor_approx=tensor_2)
    assert isinstance(residual, Tensor)
    assert (residual.mode_names == true_default_mode_names)
    np.testing.assert_array_equal(residual.data, true_residual_data)

    # ------ tests for residual tensor with the TensorCPD
    array_3d = np.array([[[100., 250., 400., 550.],
                          [250., 650., 1050., 1450.],
                          [400., 1050., 1700., 2350.]],

                         [[250., 650., 1050., 1450.],
                          [650., 1925., 3200., 4475.],
                          [1050., 3200., 5350., 7500.]]]
                        )
    true_residual_data = np.zeros(array_3d.shape)
    tensor = Tensor(array=array_3d)
    ft_shape = (2, 3, 4)    # define shape of the tensor in full form
    R = 5                   # define Kryskal rank of a tensor in CP form
    core_values = np.ones(R)
    fmat = [np.arange(orig_dim * R).reshape(orig_dim, R)
            for orig_dim in ft_shape]
    tensor_cpd = TensorCPD(fmat=fmat, core_values=core_values)
    residual = residual_tensor(tensor_orig=tensor, tensor_approx=tensor_cpd)
    assert isinstance(residual, Tensor)
    assert (residual.mode_names == true_default_mode_names)
    np.testing.assert_array_equal(residual.data, true_residual_data)

    # ------ tests for residual tensor with the TensorTKD
    array_3d = np.array([[[378,   1346,   2314,   3282,   4250],
                          [1368,   4856,   8344,  11832,  15320],
                          [2358,   8366,  14374,  20382,  26390],
                          [3348,  11876,  20404,  28932,  37460]],

                         [[1458,   5146,   8834,  12522,  16210],
                          [5112,  17944,  30776,  43608,  56440],
                          [8766,  30742,  52718,  74694,  96670],
                          [12420,  43540,  74660, 105780, 136900]],

                         [[2538,   8946,  15354,  21762,  28170],
                          [8856,  31032,  53208,  75384,  97560],
                          [15174,  53118,  91062, 129006, 166950],
                          [21492,  75204, 128916, 182628, 236340]]])
    true_residual_data = np.zeros(array_3d.shape)
    tensor = Tensor(array=array_3d)
    ft_shape = (3, 4, 5)    # define shape of the tensor in full form
    ml_rank = (2, 3, 4)     # define multi-linear rank of a tensor in Tucker form
    core_size = reduce(lambda x, y: x * y, ml_rank)
    core_values = np.arange(core_size).reshape(ml_rank)
    fmat = [np.arange(ft_shape[mode] * ml_rank[mode]).reshape(ft_shape[mode],
                                                              ml_rank[mode]) for mode in range(len(ft_shape))]
    tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)
    residual = residual_tensor(tensor_orig=tensor, tensor_approx=tensor_tkd)
    assert isinstance(residual, Tensor)
    assert (residual.mode_names == true_default_mode_names)
    np.testing.assert_array_equal(residual.data, true_residual_data)

    # ------ tests for residual tensor with the TensorTT
    array_3d = np.array([[[300, 348, 396, 444, 492, 540],
                          [354, 411, 468, 525, 582, 639],
                          [408, 474, 540, 606, 672, 738],
                          [462, 537, 612, 687, 762, 837],
                          [516, 600, 684, 768, 852, 936]],

                         [[960, 1110, 1260, 1410, 1560, 1710],
                          [1230, 1425, 1620, 1815, 2010, 2205],
                          [1500, 1740, 1980, 2220, 2460, 2700],
                          [1770, 2055, 2340, 2625, 2910, 3195],
                          [2040, 2370, 2700, 3030, 3360, 3690]],

                         [[1620, 1872, 2124, 2376, 2628, 2880],
                          [2106, 2439, 2772, 3105, 3438, 3771],
                          [2592, 3006, 3420, 3834, 4248, 4662],
                          [3078, 3573, 4068, 4563, 5058, 5553],
                          [3564, 4140, 4716, 5292, 5868, 6444]],

                         [[2280, 2634, 2988, 3342, 3696, 4050],
                          [2982, 3453, 3924, 4395, 4866, 5337],
                          [3684, 4272, 4860, 5448, 6036, 6624],
                          [4386, 5091, 5796, 6501, 7206, 7911],
                          [5088, 5910, 6732, 7554, 8376, 9198]]])
    true_residual_data = np.zeros(array_3d.shape)
    tensor = Tensor(array=array_3d)
    r1, r2 = 2, 3
    I, J, K = 4, 5, 6
    core_1 = np.arange(I * r1).reshape(I, r1)
    core_2 = np.arange(r1 * J * r2).reshape(r1, J, r2)
    core_3 = np.arange(r2 * K).reshape(r2, K)
    core_values = [core_1, core_2, core_3]
    ft_shape = (I, J, K)
    tensor_tt = TensorTT(core_values=core_values)
    residual = residual_tensor(tensor_orig=tensor, tensor_approx=tensor_tt)
    assert isinstance(residual, Tensor)
    assert (residual.mode_names == true_default_mode_names)
    np.testing.assert_array_equal(residual.data, true_residual_data)

    # ------ tests that should FAIL for residual tensor due to wrong input type
    array_3d = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7],
                          [8, 9, 10, 11]],

                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]])
    tensor_1 = Tensor(array=array_3d)
    tensor_2 = array_3d
    with pytest.raises(TypeError):
        residual_tensor(tensor_orig=tensor_1, tensor_approx=tensor_2)

    tensor_1 = array_3d
    tensor_2 = Tensor(array=array_3d)
    with pytest.raises(TypeError):
        residual_tensor(tensor_orig=tensor_1, tensor_approx=tensor_2)
