from ..synthetic import *
from ...core.structures import Tensor


#TODO: make this a more robust test
def test_make_clusters(): 
    true_dim = 3
    nsamples = 1000
    ncents = 5
    tensor = make_clusters(dims=true_dim, n_samples=nsamples, centers=ncents)
    assert isinstance(tensor, Tensor)
    assert true_dim == tensor.order
    assert tensor.shape == (nsamples, 1, true_dim)


def test_make_clusters_arrs(): 
    true_dim = 3
    nsamples = [200]*5
    ncents = 5
    cents = np.random.uniform(size=(ncents, 1, true_dim))
    print(cents)
    tensor, centers = make_clusters(dims=true_dim, n_samples=nsamples, centers=cents, return_centers=True)
    assert isinstance(tensor, Tensor)
    assert true_dim == tensor.order
    assert tensor.shape == (1000, 1, true_dim)
    cents, centers = np.asarray(cents), np.asarray(cents)
    assert np.array_equal(cents, centers)
