from ..matrices import *
from ....core.structures import Tensor, TensorCPD, TensorTKD, TensorTT

def test_genToeplitzMatrix():
    mat = toeplitz_matrix([1, 2, 3, 4, 5, 6], [1, 4, 5, 6])

    true_mat = np.array([[1, 2, 3, 4, 5, 6],
                         [4, 1, 2, 3, 4, 5],
                         [5, 4, 1, 2, 3, 4],
                         [6, 5, 4, 1, 2, 3]])
    assert np.array_equal(true_mat, mat)

def test_genHankelMatrix():
    mat = hankel_matrix([1, 2, 3, 4, 5, 6], [1, 4, 5, 6])

    true_mat = np.array([[1, 4, 5, 6, 5, 4],
                         [4, 5, 6, 5, 4, 3],
                         [5, 6, 5, 4, 3, 2],
                         [6, 5, 4, 3, 2, 1]])
    assert np.array_equal(true_mat, mat)
