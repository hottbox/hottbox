import pytest
import sys
import io
import numpy as np
from functools import reduce
from collections import OrderedDict
from ..structures import *


class TestTensor:
    """ Tests for Tensor class """

    def test_init(self):
        """ Tests for Tensor object creation """
        # ------ tests that Tensor object can be created only from numpy array
        array = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(TypeError):
            Tensor(array=array)
        with pytest.raises(TypeError):
            Tensor(array=Tensor(array))

        # ------ tests for basic properties of a tensor with default mode names
        true_shape = (2, 4, 8)
        true_size = reduce(lambda x, y: x * y, true_shape)
        true_order = len(true_shape)
        true_data = np.ones(true_size).reshape(true_shape)
        true_default_mode_names = OrderedDict([(0, 'mode-0'),
                                               (1, 'mode-1'),
                                               (2, 'mode-2')
                                               ])
        tensor = Tensor(array=true_data)
        np.testing.assert_array_equal(tensor.data, true_data)
        assert (tensor.frob_norm == 8.0)
        assert (tensor.shape == true_shape)
        assert (tensor.ft_shape == true_shape)
        assert (tensor.order == true_order)
        assert (tensor.size == true_size)
        assert (tensor.mode_names == true_default_mode_names)
        assert (tensor._data is not true_data)          # check that is not a reference

        # ------ tests for creating a Tensor object with custom mode names
        true_custom_mode_names = OrderedDict([(0, 'time'),
                                              (1, 'frequency'),
                                              (2, 'channel')
                                              ])
        tensor = Tensor(array=true_data, mode_names=true_custom_mode_names)
        assert (tensor.mode_names == true_custom_mode_names)        # check that values are the same
        assert (tensor._mode_names is not true_custom_mode_names)   # check that not a reference

        # ------ tests that should FAIL for custom mode names being incorrectly defined
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
            incorrect_mode_names = OrderedDict(
                [(mode, "{}-mode".format(mode)) for mode in range(true_order - 2, true_order + 1)])
            Tensor(array=true_data, mode_names=incorrect_mode_names)

        with pytest.raises(ValueError):
            # key value is set to be negative
            incorrect_mode_names = OrderedDict([(mode, "{}-mode".format(mode)) for mode in range(-1, true_order - 1)])
            Tensor(array=true_data, mode_names=incorrect_mode_names)

        # ------ tests for creating a Tensor object with custom ft_shape
        I, J, K = 2, 4, 8
        true_data = np.ones(I * J * K).reshape(I, J, K)
        true_ft_shape = (I, J, K)
        tensor = Tensor(array=true_data, ft_shape=true_ft_shape)
        assert (tensor.ft_shape == true_ft_shape)       # check that values are the same
        assert (tensor._ft_shape is not true_ft_shape)  # check that not a reference

        # check when ft_shape is correct but do not correspond to the shape of data array
        I, J, K = 2, 4, 8
        true_data = np.ones(I * J * K).reshape(I, J*K)
        true_ft_shape = (I, J, K)
        tensor = Tensor(array=true_data, ft_shape=true_ft_shape)
        assert (tensor.ft_shape == true_ft_shape)  # check that values are the same
        assert (tensor._ft_shape is not true_ft_shape)  # check that not a reference

        # ------ tests that should FAIL for custom ft_shape being incorrectly defined
        with pytest.raises(TypeError):
            # should of of tuple type
            incorrect_ft_shape = list(true_shape)
            Tensor(array=true_data, ft_shape=incorrect_ft_shape)

        with pytest.raises(ValueError):
            # shape does not match the number of elements
            I, J, K = 2, 4, 8
            true_data = np.ones(I*J*K).reshape(I, J, K)
            incorrect_ft_shape = (I+1, J, K)
            Tensor(array=true_data, ft_shape=incorrect_ft_shape)

        with pytest.raises(ValueError):
            # shape does not match the number of elements
            I, J, K = 2, 4, 8
            true_data = np.ones(I*J*K).reshape(I, J*K)
            incorrect_ft_shape = (I+1, J, K)
            Tensor(array=true_data, ft_shape=incorrect_ft_shape)

    def test_copy(self):
        """ Tests for creation a copy of a Tensor object """
        data = np.arange(24).reshape(2, 3, 4)
        tensor = Tensor(data)
        tensor_copy = tensor.copy()
        assert (tensor_copy is not tensor)
        assert (tensor_copy._data is not tensor._data)
        assert (tensor_copy._ft_shape is not tensor._ft_shape)
        assert (tensor_copy._mode_names is not tensor._mode_names)
        np.testing.assert_array_equal(tensor_copy.data, tensor.data)
        assert (tensor_copy.ft_shape == tensor.ft_shape)
        assert (tensor_copy.frob_norm == tensor.frob_norm)
        assert (tensor_copy.shape == tensor.shape)
        assert (tensor_copy.order == tensor.order)
        assert (tensor_copy.size == tensor.size)
        assert (tensor_copy.mode_names == tensor.mode_names)

    def test_rename_modes(self):
        """ Tests for renaming modes """
        true_shape = (2, 4, 8)
        true_size = reduce(lambda x, y: x * y, true_shape)
        true_order = len(true_shape)
        true_data = np.ones(true_size).reshape(true_shape)
        orig_mode_names = OrderedDict([(0, '1-mode'),
                                       (1, '2-mode'),
                                       (2, '3-mode')
                                       ])
        true_new_mode_names_ordered_dict = OrderedDict([(0, 'time'),
                                                        (1, 'frequency'),
                                                        (2, 'channel')
                                                        ])
        true_new_mode_names_dict = {0: 'pixel_x',
                                    1: 'pixel_y',
                                    2: 'color'
                                    }
        tensor = Tensor(array=true_data, mode_names=orig_mode_names)

        tensor.rename_modes(true_new_mode_names_ordered_dict)
        assert (tensor.mode_names == true_new_mode_names_ordered_dict)  # check that values are the same
        assert (tensor._mode_names is not true_new_mode_names_ordered_dict)  # check that not a reference

        # test that it also works for dict
        tensor.rename_modes(true_new_mode_names_dict)
        assert (tensor.mode_names == true_new_mode_names_dict)  # check that values are the same
        assert (tensor._mode_names is not true_new_mode_names_dict)  # check that not a reference

        # ------ tests that should FAIL for new mode names being incorrectly defined for renaming
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

    def test_describe(self):
        """ Tests for describe function of a Tensor object """
        # TODO: find a better way to test the method that only prints
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output     # and redirect stdout.

        true_shape = (2, 4, 8)
        true_order = len(true_shape)
        true_size = reduce(lambda x, y: x * y, true_shape)
        true_data = np.ones(true_size).reshape(true_shape)
        true_mode_names = OrderedDict([(0, 'time'),
                                       (1, 'frequency'),
                                       (2, 'channel')
                                       ])
        tensor = Tensor(array=true_data, mode_names=true_mode_names)
        tensor.describe()
        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # check that this function does not change anything in the object
        np.testing.assert_array_equal(tensor.data, true_data)
        assert (tensor.frob_norm == 8.0)
        assert (tensor.shape == true_shape)
        assert (tensor.ft_shape == true_shape)
        assert (tensor.order == true_order)
        assert (tensor.size == true_size)
        assert (tensor.mode_names == true_mode_names)

    def test_fold(self):
        """ Tests for folding a Tensor object """
        true_folded_shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, true_folded_shape)
        true_result_data = np.arange(size).reshape(true_folded_shape)
        true_result_mode_names = OrderedDict([(0, 'time'),
                                              (1, 'frequency'),
                                              (2, 'person')
                                              ])
        orig_unfolded_0_mode_names = OrderedDict([(0, OrderedDict([(0, 'time')])),
                                                  (1, OrderedDict([(1, 'frequency'), (2, 'person')]))
                                                  ])
        orig_unfolded_1_mode_names = OrderedDict([(0, OrderedDict([(1, 'frequency')])),
                                                  (1, OrderedDict([(0, 'time'), (2, 'person')]))
                                                  ])
        orig_unfolded_2_mode_names = OrderedDict([(0, OrderedDict([(2, 'person')])),
                                                  (1, OrderedDict([(0, 'time'), (1, 'frequency')]))
                                                  ])
        orig_unfolded_mode_names = [orig_unfolded_0_mode_names, orig_unfolded_1_mode_names, orig_unfolded_2_mode_names]
        orig_unfolded_data = [unfold(tensor=true_result_data, mode=mode) for mode in range(len(true_folded_shape))]

        # check that there would be no changes if tensor hasn't been unfolded previously
        tensor = Tensor(array=true_result_data, mode_names=true_result_mode_names)
        tensor.fold(inplace=True)
        np.testing.assert_array_equal(tensor.data, true_result_data)
        assert (tensor.mode_names == true_result_mode_names)

        # --------- tests for folding INPLACE=TRUE
        for mode in range(len(true_folded_shape)):
            tensor = Tensor(array=orig_unfolded_data[mode],
                            mode_names=orig_unfolded_mode_names[mode],
                            ft_shape=true_folded_shape)
            tensor.fold(inplace=True)
            np.testing.assert_array_equal(tensor.data, true_result_data)
            assert (tensor.mode_names == true_result_mode_names)

        # --------- tests for unfolding INPLACE=FALSE
        for mode in range(len(true_folded_shape)):
            tensor = Tensor(array=orig_unfolded_data[mode],
                            mode_names=orig_unfolded_mode_names[mode],
                            ft_shape=true_folded_shape)
            tensor_folded = tensor.fold(inplace=False)

            # check that a new object was returned, but values `_ft_shape` were preserved
            assert (tensor_folded is not tensor)
            assert (tensor_folded._ft_shape is not tensor._ft_shape)
            assert (tensor_folded.ft_shape == tensor.ft_shape)

            # check that the original tensor object has not been modified
            # (no internal references with the unfolded version)
            np.testing.assert_array_equal(tensor.data, orig_unfolded_data[mode])
            assert (tensor.mode_names == orig_unfolded_mode_names[mode])

            # check the result of folding
            np.testing.assert_array_equal(tensor_folded.data, true_result_data)
            assert (tensor_folded.mode_names == true_result_mode_names)

    def test_unfold(self):
        """ Tests for unfolding a Tensor object """
        true_orig_shape = (2, 2, 2)
        size = reduce(lambda x, y: x * y, true_orig_shape)
        orig_folded_data = np.arange(size).reshape(true_orig_shape)
        orig_folded_mode_names = OrderedDict([(0, 'time'),
                                              (1, 'frequency'),
                                              (2, 'person')
                                              ])
        true_result_0_mode_names = OrderedDict([(0, OrderedDict([(0, 'time')])),
                                                (1, OrderedDict([(1, 'frequency'), (2, 'person')]))
                                                ])
        true_result_1_mode_names = OrderedDict([(0, OrderedDict([(1, 'frequency')])),
                                                (1, OrderedDict([(0, 'time'), (2, 'person')]))
                                                ])
        true_result_2_mode_names = OrderedDict([(0, OrderedDict([(2, 'person')])),
                                                (1, OrderedDict([(0, 'time'), (1, 'frequency')]))
                                                ])
        true_result_mode_names = [true_result_0_mode_names, true_result_1_mode_names, true_result_2_mode_names]
        true_result_data = [unfold(tensor=orig_folded_data, mode=mode) for mode in range(orig_folded_data.ndim)]

        # --------- tests for unfolding INPLACE=TRUE
        for mode in range(orig_folded_data.ndim):
            tensor = Tensor(array=orig_folded_data, mode_names=orig_folded_mode_names)
            tensor.unfold(mode=mode, inplace=True)
            np.testing.assert_array_equal(tensor.data, true_result_data[mode])
            assert (tensor.mode_names == true_result_mode_names[mode])

        # --------- tests for unfolding INPLACE=FALSE
        for mode in range(orig_folded_data.ndim):
            tensor = Tensor(array=orig_folded_data, mode_names=orig_folded_mode_names)
            tensor_unfolded = tensor.unfold(mode=mode, inplace=False)
            # check that a new object was returned, but values `_ft_shape` were preserved
            assert (tensor_unfolded is not tensor)
            assert (tensor_unfolded._ft_shape is not tensor._ft_shape)
            assert (tensor_unfolded._ft_shape == tensor._ft_shape)

            # check that the original tensor object has not been modified
            # (no internal references with the unfolded version)
            np.testing.assert_array_equal(tensor.data, orig_folded_data)
            assert (tensor.mode_names == orig_folded_mode_names)

            # check the result of unfolding
            np.testing.assert_array_equal(tensor_unfolded.data, true_result_data[mode])
            assert (tensor_unfolded.mode_names == true_result_mode_names[mode])

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
            assert (tensor_res._ft_shape is not tensor._ft_shape)
            np.testing.assert_array_equal(tensor_res.data, true_res)
            # check that the original tensor object has not been modified
            np.testing.assert_array_equal(tensor.data, array_3d)

        # ------  test for changing mode_names correctly
        orig_dim = (5, 6, 7)
        new_dim = [2, 3, 4]
        size = reduce(lambda x, y: x * y, orig_dim)
        array_3d = np.arange(size).reshape(orig_dim)
        orig_names = OrderedDict([(0, 'country'),
                                  (1, 'model'),
                                  (2, 'year')
                                  ])

        # check that names have not been changed when multiply with numpy array
        for mode in range(len(new_dim)):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode])
            tensor.mode_n_product(matrix, mode=mode)
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
        new_matrix_name = {0 : 'age'}
        for mode in range(len(new_dim)):
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = Tensor(np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode]))
            matrix.rename_modes(new_mode_names=new_matrix_name)
            tensor.mode_n_product(matrix, mode=mode)
            new_true_names = orig_names.copy()
            new_true_names[mode] = new_matrix_name[0]
            assert (tensor.mode_names == new_true_names)

        # check that you cannot use matrix of Tensor class and specify new name at the same time
        with pytest.raises(ValueError):
            mode = 1
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = Tensor(np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode]))
            new_name = 'age'
            tensor.mode_n_product(matrix, mode=mode, new_name=new_name)

        # check that new_name should be of string type
        with pytest.raises(TypeError):
            mode = 1
            tensor = Tensor(array=array_3d, mode_names=orig_names)
            matrix = np.arange(new_dim[mode] * orig_dim[mode]).reshape(new_dim[mode], orig_dim[mode])
            new_name = 5
            tensor.mode_n_product(matrix, mode=mode, new_name=new_name)


def test_super_diag_tensor():
    """ Tests for creating super-diagonal tensor"""
    pass


def test_residual_tensor():
    """ Tests for computing/creating a residual tensor """
    pass