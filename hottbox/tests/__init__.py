from ..version import __version__


def test_version():
    """ Dummy test for version """
    assert isinstance(__version__, str)
