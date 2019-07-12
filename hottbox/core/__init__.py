"""
This module contains methods for the most common operations within multi-linear algebra and
classes for the tensors represented through various tensor decompositions
"""

from .structures import Tensor, TensorCPD, TensorTKD, TensorTT
from .operations import khatri_rao, hadamard, kronecker, mode_n_product, unfold, fold, kolda_unfold, kolda_fold, \
    sampled_khatri_rao


__all__ = [
    "Tensor",
    "TensorCPD",
    "TensorTKD",
    "TensorTT",
    "khatri_rao",
    "sampled_khatri_rao",
    "hadamard",
    "kronecker",
    "mode_n_product",
    "unfold",
    "fold",
    "kolda_unfold",
    "kolda_fold",
]
