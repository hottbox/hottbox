"""
Tests for the metrics base module

Notes
-----
No need to test the wrong input type,
since it is taken care of in computing residual tensor
"""
import numpy as np
from functools import reduce
from ..base import *
from ...core.structures import Tensor


def test_mse():
    """ Tests for mse """
    # ------ tests for 1-d case
    shape = (2,)
    size = shape[0]
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mse = 0.5
    result = mse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mse)

    # ------ tests for 2-d case
    shape = (2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mse = 3.5
    result = mse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mse)

    # ------ tests for 3-d case
    shape = (2, 2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mse = 17.5
    result = mse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mse)


def test_rmse():
    """ Tests for rmse """
    # ------ tests for 1-d case
    shape = (2,)
    size = shape[0]
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_rmse = 0.7071067811865476
    result = rmse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_rmse)

    # ------ tests for 2-d case
    shape = (2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_rmse = 1.8708286933869707
    result = rmse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_rmse)

    # ------ tests for 3-d case
    shape = (2, 2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_rmse = 4.183300132670378
    result = rmse(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_rmse)


def test_mape():
    """ Tests for mape """
    # ------ tests for 1-d case
    shape = (2,)
    size = shape[0]
    tensor_true = Tensor(np.arange(size).reshape(shape) + 1)
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mape = 0.5
    result = mape(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mape)

    # ------ tests for 2-d case
    shape = (2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape) + 1)
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mape = 0.4583333333333333
    result = mape(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mape)

    # ------ tests for 3-d case
    shape = (2, 2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape) + 1)
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_mape = 0.5705357142857143
    result = mape(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_mape)


def test_residual_rel_error():
    # ------ tests for 1-d case
    shape = (2,)
    size = shape[0]
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_res_rel_error = 1
    result = residual_rel_error(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_res_rel_error)

    # ------ tests for 2-d case
    shape = (2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_res_rel_error = 1
    result = residual_rel_error(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_res_rel_error)

    # ------ tests for 3-d case
    shape = (2, 2, 2)
    size = reduce(lambda x, y: x * y, shape)
    tensor_true = Tensor(np.arange(size).reshape(shape))
    tensor_pred = Tensor(np.arange(size).reshape(shape) * 2)
    true_res_rel_error = 1
    result = residual_rel_error(tensor_true, tensor_pred)
    np.testing.assert_array_almost_equal(result, true_res_rel_error)
