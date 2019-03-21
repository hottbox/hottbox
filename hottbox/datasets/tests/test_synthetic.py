import pytest
import numpy as np
from ..synthetic import *
from ...core.structures import Tensor
from ...utils.checks import is_super_symmetric

#TODO: make this a more robust test
def test_make_clusters(): 
    true_dim = 3
    nsamples = 1000
    ncents = 5
    tensor = make_clusters(dims=true_dim, n_samples=nsamples, centers=ncents)
    assert isinstance(tensor, Tensor)
    assert true_dim == tensor.order
    assert tensor.shape == (nsamples, 1, true_dim) 
