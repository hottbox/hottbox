"""
Module for various algorithms of tensor decompositions
"""

from .cpd import CPD, RandomisedCPD
from .tucker import HOSVD, HOOI
from .tensor_train import TTSVD


__all__ = [
    "CPD", "RandomisedCPD",
    "HOSVD", "HOOI",
    "TTSVD", "Parafac2"
]
