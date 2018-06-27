import numpy as np
from math import sqrt
from ..core.structures import Tensor, TensorCPD, TensorTKD, TensorTT, residual_tensor


def mse(tensor_true, tensor_pred):
    """ Mean squared error

    Parameters
    ----------
    tensor_true : Tensor
    tensor_pred : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    float
    """
    tensor_res = residual_tensor(tensor_true, tensor_pred)
    return np.mean(tensor_res.data ** 2)


def rmse(tensor_true, tensor_pred):
    """ Root mean squared error

    Parameters
    ----------
    tensor_true : Tensor
    tensor_pred : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    float
    """
    return sqrt(mse(tensor_true, tensor_pred))


def mape(tensor_true, tensor_pred):
    """ Mean absolute percentage error

    Parameters
    ----------
    tensor_true : Tensor
    tensor_pred : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    float
    """
    # TODO: Fix cases when tensor_pred.data contains zeros (division by zero -> inf)
    tensor_res = residual_tensor(tensor_true, tensor_pred)
    return np.mean(np.abs(np.divide(tensor_res.data, tensor_true.data)))


def residual_rel_error(tensor_true, tensor_pred):
    """ Relative error of approximation

    Parameters
    ----------
    tensor_true : Tensor
    tensor_pred : {Tensor, TensorCPD, TensorTKD, TensorTT}

    Returns
    -------
    float
    """
    tensor_res = residual_tensor(tensor_true, tensor_pred)
    return tensor_res.frob_norm / tensor_true.frob_norm
