"""
This module contains various models for classification
"""

from .ensemble_learning import TelVI, TelVAC
from .stm import LSSTM


__all__ = [
    "LSSTM",
    "TelVI",
    "TelVAC",
]
