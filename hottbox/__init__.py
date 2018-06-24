import importlib


_PERFORM_VALIDATION = True


def perform_input_validation(validate):
    """ Global settings for all classes

    Parameters
    ----------
    validate : bool
    """
    from .core import structures
    global _PERFORM_VALIDATION
    _PERFORM_VALIDATION = validate
    importlib.reload(structures)
