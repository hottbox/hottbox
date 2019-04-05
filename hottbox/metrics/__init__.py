"""
This module includes score functions and performance metrics for the
computational methods defined in ``hottbox.algorithms``
"""

from .decomposition import mse, rmse, mape, residual_rel_error


__all__ = [
    "mse",
    "rmse",
    "mape",
    "residual_rel_error",
]
