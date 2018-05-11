from .. import version
from ..utils import *

def test_version():
    assert isinstance(version.__version__, str)