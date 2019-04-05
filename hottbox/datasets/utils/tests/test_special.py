import numpy as np
from ..special import toeplitz
from ....utils.checks import is_toep_tensor
from scipy.linalg import toeplitz as toeplitz_mat


def test_toeplitz():
    tensor = np.zeros(shape=(4,4,3))
    # Inititalise
    mat_A = toeplitz_mat(r=[1,2,3,4], c=[1,4,3,2])
    mat_B = toeplitz_mat(r=[13,5,17,8], c=[13,18,17,5])
    mat_C = toeplitz_mat(r=[0,9,30,11], c=[0,11,30,9])
    tensor[:,:,0] = mat_A
    tensor[:,:,1] = mat_B
    tensor[:,:,2] = mat_C

    tt = toeplitz((4,4,3), matC=np.array([mat_A, mat_B, mat_C])).data
    assert np.array_equal(tt, tensor)


def test_toeplitz_random():
    test_tensor = toeplitz((3,3,4), modes=[0,1], random=True)
    assert is_toep_tensor(test_tensor, modes=[0,1])
