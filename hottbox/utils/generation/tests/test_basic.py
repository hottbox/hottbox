import pytest
import numpy as np
from functools import reduce
from hottbox.core.structures import Tensor,TensorCPD, TensorTKD, TensorTT
from hottbox.utils.validation.checks import is_super_symmetric
from ..basic import dense_tensor, sparse_tensor, super_diagonal_tensor, \
    super_diag_tensor, super_symmetric_tensor, residual_tensor


def test_dense():
    # TODO: test distribution
    true_shape = (4,3,2,3)
    true_distribution = 'normal'
    true_type = 0
    tensor = dense_tensor(true_shape, true_distribution, true_type)
    assert isinstance(tensor, Tensor)
    assert true_shape == tensor.shape
    pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
    assert pct_nonzero > 0.8


def test_sparse():
    true_shape = (4,3,2,3)
    true_distribution = 'normal'
    true_type = 0
    tensor = sparse_tensor(true_shape, true_distribution, true_type)
    assert isinstance(tensor, Tensor)
    assert true_shape == tensor.shape
    pct_nonzero = np.count_nonzero(tensor.data) / tensor.data.size
    assert pct_nonzero < 0.08 and pct_nonzero > 0.02


def test_superdiagonal():
    true_shape = (4,4,4)
    true_distribution = 'ones'
    true_type = 0
    tensor = super_diagonal_tensor(true_shape, true_distribution)
    assert true_shape == tensor.shape
    assert isinstance(tensor, Tensor)
    tensor = tensor.data
    trace = 0
    for i in range(true_shape[0]):
        trace += tensor[i,i,i]
    assert trace == true_shape[0]


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


def test_supersymmetric():
    true_shape = (4,4,4)
    tensor = super_symmetric_tensor(true_shape)

    assert true_shape == tensor.shape
    assert isinstance(tensor, Tensor)
    assert is_super_symmetric(tensor)


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
