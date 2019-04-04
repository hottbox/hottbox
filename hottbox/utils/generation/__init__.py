"""
Module for generation of synthetic N-dimensional arrays (tensors)
"""
from .matrices import toeplitz_matrix, hankel_matrix
from .tensors import quick_tensor, quick_tensorcpd, quick_tensortkd, quick_tensortt
from .basic import dense_tensor, super_diagonal_tensor, super_symmetric_tensor, \
    sparse_tensor, super_diag_tensor, residual_tensor
from .special import toeplitz_tensor


__all__ = [
    "toeplitz_matrix",
    "hankel_matrix",
    "quick_tensor",
    "quick_tensorcpd",
    "quick_tensortkd",
    "quick_tensortt",
    "dense_tensor",
    "super_diagonal_tensor",
    "super_symmetric_tensor",
    "sparse_tensor",
    "super_diag_tensor",
    "residual_tensor",
    "toeplitz_tensor",
]
