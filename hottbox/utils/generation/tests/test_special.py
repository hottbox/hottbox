import numpy as np
from scipy.linalg import toeplitz as toeplitz_mat
from hottbox.utils.generation.special import toeplitz_tensor
from hottbox.utils.validation.checks import is_toeplitz_tensor


def test_toeplitz():
    tensor = np.zeros(shape=(4,4,3))
    # Inititalise
    mat_A = toeplitz_mat(r=[1, 2, 3, 4], c=[1, 4, 3, 2])
    mat_B = toeplitz_mat(r=[13, 5, 17, 8], c=[13, 18, 17, 5])
    mat_C = toeplitz_mat(r=[0, 9, 30, 11], c=[0, 11, 30, 9])
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C

    tt = toeplitz_tensor((4, 4, 3), matC=np.array([mat_A, mat_B, mat_C])).data
    assert np.array_equal(tt, tensor)


def test_toeplitz_random():
    test_tensor = toeplitz_tensor((3, 3, 4), modes=[0, 1], random=True)
    assert is_toeplitz_tensor(test_tensor, modes=[0, 1])
