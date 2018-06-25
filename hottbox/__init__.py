import importlib


_PERFORM_VALIDATION = True


def perform_input_validation(validate):
    """ Global settings for all classes

    Parameters
    ----------
    validate : bool

    Notes
    -----
    Reloading of modules that include classes with inheritance results in run time error
    ::
        TypeError: super(type, obj): obj must be an instance or subtype of type

    This issue is described
    `here1 <http://thomas-cokelaer.info/blog/2011/09/382/>`_ and
    `here2 <https://thingspython.wordpress.com/2010/09/27/another-super-wrinkle-raising-typeerror/>`_
    """
    # TODO: how is about to set environmental variable instead of reloading modules?
    from .core import structures
    global _PERFORM_VALIDATION
    _PERFORM_VALIDATION = validate
    importlib.reload(structures)
