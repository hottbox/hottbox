"""
Module for generation of synthetic N-dimensional arrays (tensors)
"""
from .matrices import toeplitz_matrix, hankel_matrix
from .tensors import quick_tensor, quick_tensorcpd, quick_tensortkd, quick_tensortt
from .basic import dense_tensor, super_diagonal_tensor, super_symmetric_tensor