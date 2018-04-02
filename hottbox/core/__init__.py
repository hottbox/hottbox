"""
This module includes methods for the most common operations within multilinear algebra and
classes that can represent tensors through various tensor decompositions
"""

from .structures import Tensor, TensorCPD, TensorTKD, TensorTT, super_diag_tensor
from .operations import khatri_rao, hadamard, kronecker, mode_n_product, unfold, fold