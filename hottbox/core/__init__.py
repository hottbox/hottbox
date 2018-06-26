"""
This module contains methods for the most common operations within multilinear algebra and
classes for the tensors represented through various tensor decompositions
"""

from .structures import Tensor, TensorCPD, TensorTKD, TensorTT, super_diag_tensor, residual_tensor
from .operations import khatri_rao, hadamard, kronecker, mode_n_product, unfold, fold, kolda_unfold, kolda_fold