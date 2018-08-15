"""
This module includes all custom warnings and errors used across ``hottbox``.
"""


class TensorStateError(Exception):
    """
    Error raised when attempting to perform an operation on a ``Tensor``
    which is not allowed for its current state
    """
    pass


class TensorModeError(Exception):
    """
    Error raised when attempting to perform an operation on a ``Tensor``
    which is not allowed by its ``Mode``
    """
    pass


class TensorShapeError(Exception):
    """
    Error raised when attempting to perform an operation on a ``Tensor``
    which is not allowed for its current shape
    """
    pass


class TensorTopologyError(Exception):
    """
    Error related to the dimensionality mismatch of counterparts of
    ``TensorCPD``, ``TensorTKD`` and ``TensorTT``
    """
    pass


class StateError(Exception):
    """
    Error raised when there is an attempt to set
    incorrect parameters for state of a ``Tensor``
    """
    pass


class ModeError(Exception):
    """
    Error raised when there is an attempt to set
    incorrect parameters for mode of a ``Tensor``
    """
    pass
