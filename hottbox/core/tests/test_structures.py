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


class TestBaseTensorTD:
    """ Tests for the BaseTensorTD as an interface"""

    def test_init(self):
        tensor_interface = BaseTensorTD()
        with pytest.raises(NotImplementedError):
            tensor_interface.copy()
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
            tensor_interface.reconstruct


class TestTensorCPD:
    """ Tests for the TensorCPD class """

    def test_init(self):
        """ Tests for the TensorCPD constructor """
        ft_shape = (3, 4, 5)    # define shape of the tensor in full form
        R = 2                   # define Kryskal rank of a tensor in CP form
        core_values = np.ones(R)
        true_orig_fmat_list = [np.arange(orig_dim * R).reshape(orig_dim, R) for orig_dim in ft_shape]
        fmat_list = [fmat.copy() for fmat in true_orig_fmat_list]

        tensor_cpd = TensorCPD(fmat=fmat_list, core_values=core_values)
        assert isinstance(tensor_cpd.fmat, list)
        assert tensor_cpd.order == len(fmat_list)
        assert isinstance(tensor_cpd.rank, tuple)
        assert tensor_cpd.rank == (R,)
        assert isinstance(tensor_cpd._core_values, np.ndarray)
        np.testing.assert_array_equal(tensor_cpd._core_values, core_values)
        assert tensor_cpd._core_values is not core_values


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
        with pytest.raises(ValueError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = np.ones([2,2,2])
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

        # too many (or not enough) `core_values` for `fmat`
        with pytest.raises(ValueError):
            incorrect_core_values = np.ones(correct_core_values.size + 1)
            TensorCPD(fmat=correct_fmat, core_values=incorrect_core_values)

        # dimension all factor matrices should have the same number of columns
        with pytest.raises(ValueError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = incorrect_fmat[0].T
            TensorCPD(fmat=incorrect_fmat, core_values=correct_core_values)

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
        true_default_mode_names = OrderedDict([(0, 'mode-0'),
                                               (1, 'mode-1'),
                                               (2, 'mode-2')
                                               ])
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
        tensor_rec = tensor_cpd.reconstruct
        assert isinstance(tensor_rec, Tensor)
        np.testing.assert_array_equal(tensor_rec.data, true_data)
        assert (tensor_rec.ft_shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for consecutive reconstructions: results should be the same but different objects
        tensor_rec_1 = tensor_cpd.reconstruct
        tensor_rec_2 = tensor_cpd.reconstruct
        np.testing.assert_array_equal(tensor_rec_1.data, true_data)
        np.testing.assert_array_equal(tensor_rec_1.data, tensor_rec_2.data)
        assert tensor_rec_1 is not tensor_rec_2

        # ------ tests for chaining methods
        new_mode_names =  OrderedDict([(0, 'frequency'),
                                       (1, 'time'),
                                       (2, 'channel')
                                       ])
        mode = 0
        new_dim_size = 7
        matrix = np.arange(new_dim_size * ft_shape[mode]).reshape(new_dim_size, ft_shape[mode])

        tensor_rec = tensor_cpd.reconstruct.rename_modes(new_mode_names=new_mode_names)
        assert (tensor_rec.mode_names == new_mode_names)

        new_name = 'age'
        tensor_rec = tensor_cpd.reconstruct.mode_n_product(matrix, mode=mode, new_name=new_name)
        new_shape = [i for i in ft_shape]
        new_shape[mode] = new_dim_size
        new_shape = tuple(new_shape)
        new_mode_names = true_default_mode_names
        new_mode_names[mode] = new_name
        assert (tensor_rec.shape == new_shape)
        assert (tensor_rec.mode_names == new_mode_names)


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
        with pytest.raises(ValueError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            mode = 0
            incorrect_fmat[mode] = np.ones([ft_shape[mode], ml_rank[mode], 2])
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

        # Not enough factor matrices for the specified core tensor
        with pytest.raises(ValueError):
            incorrect_core_values = np.ones(correct_core_values.shape + (2,))
            TensorTKD(fmat=correct_fmat, core_values=incorrect_core_values)

        # number of columns of some factor matrices does not match the size of the corresponding mode of the core
        with pytest.raises(ValueError):
            incorrect_fmat = [fmat.copy() for fmat in correct_fmat]
            incorrect_fmat[0] = incorrect_fmat[0].T
            TensorTKD(fmat=incorrect_fmat, core_values=correct_core_values)

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
        true_default_mode_names = OrderedDict([(0, 'mode-0'),
                                               (1, 'mode-1'),
                                               (2, 'mode-2')
                                               ])
        true_data = np.array([[[  491400,  1628200,  2765000,  3901800],
                               [ 1609020,  5330080,  9051140, 12772200],
                               [ 2726640,  9031960, 15337280, 21642600]],

                              [[ 1389150,  4596200,  7803250, 11010300],
                               [ 4507020, 14906780, 25306540, 35706300],
                               [ 7624890, 25217360, 42809830, 60402300]]]
                             )

        tensor_tkd = TensorTKD(fmat=fmat, core_values=core_values)

        # ------ basic tests on getting correct results after reconstruction
        tensor_rec = tensor_tkd.reconstruct
        assert isinstance(tensor_rec, Tensor)
        np.testing.assert_array_equal(tensor_rec.data, true_data)
        assert (tensor_rec.ft_shape == ft_shape)
        assert (tensor_rec.mode_names == true_default_mode_names)

        # ------ tests for consecutive reconstructions: results should be the same but different objects
        tensor_rec_1 = tensor_tkd.reconstruct
        tensor_rec_2 = tensor_tkd.reconstruct
        np.testing.assert_array_equal(tensor_rec_1.data, true_data)
        np.testing.assert_array_equal(tensor_rec_1.data, tensor_rec_2.data)
        assert tensor_rec_1 is not tensor_rec_2

        # ------ tests for chaining methods
        new_mode_names = OrderedDict([(0, 'frequency'),
                                      (1, 'time'),
                                      (2, 'channel')
                                      ])
        mode = 0
        new_dim_size = 7
        matrix = np.arange(new_dim_size * ft_shape[mode]).reshape(new_dim_size, ft_shape[mode])

        tensor_rec = tensor_tkd.reconstruct.rename_modes(new_mode_names=new_mode_names)
        assert (tensor_rec.mode_names == new_mode_names)

        new_name = 'age'
        tensor_rec = tensor_tkd.reconstruct.mode_n_product(matrix, mode=mode, new_name=new_name)
        new_shape = [i for i in ft_shape]
        new_shape[mode] = new_dim_size
        new_shape = tuple(new_shape)
        new_mode_names = true_default_mode_names
        new_mode_names[mode] = new_name
        assert (tensor_rec.shape == new_shape)
        assert (tensor_rec.mode_names == new_mode_names)


class TestTensorTT:
    pass


def test_super_diag_tensor():
    """ Tests for creating super-diagonal tensor"""
    pass


def test_residual_tensor():
    """ Tests for computing/creating a residual tensor """
    pass