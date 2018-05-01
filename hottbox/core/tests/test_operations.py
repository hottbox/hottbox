"""
Tests for the operations module
"""
import numpy as np
from functools import reduce
from ..operations import *

def test_mode_n_product():
    """Tests for the mode-n product"""

    I, J, K = 2 ,3, 4
    I_new, J_new, K_new = 5, 6, 7
    array_3d = np.arange(I*J*K).reshape((I, J, K))
    A = np.arange(I_new * I).reshape(I_new, I)
    B = np.arange(J_new * J).reshape(J_new, J)
    C = np.arange(K_new * K).reshape(K_new, K)

    true_res_0 = np.array([[[ 12,  13,  14,  15],
                            [ 16,  17,  18,  19],
                            [ 20,  21,  22,  23]],

                           [[ 36,  41,  46,  51],
                            [ 56,  61,  66,  71],
                            [ 76,  81,  86,  91]],

                           [[ 60,  69,  78,  87],
                            [ 96, 105, 114, 123],
                            [132, 141, 150, 159]],

                           [[ 84,  97, 110, 123],
                            [136, 149, 162, 175],
                            [188, 201, 214, 227]],

                           [[108, 125, 142, 159],
                            [176, 193, 210, 227],
                            [244, 261, 278, 295]]])

    true_res_1 = np.array([[[ 20,  23,  26,  29],
                            [ 56,  68,  80,  92],
                            [ 92, 113, 134, 155],
                            [128, 158, 188, 218],
                            [164, 203, 242, 281],
                            [200, 248, 296, 344]],

                           [[ 56,  59,  62,  65],
                            [200, 212, 224, 236],
                            [344, 365, 386, 407],
                            [488, 518, 548, 578],
                            [632, 671, 710, 749],
                            [776, 824, 872, 920]]])

    true_res_2 = np.array([[[  14,   38,   62,   86,  110,  134,  158],
                            [  38,  126,  214,  302,  390,  478,  566],
                            [  62,  214,  366,  518,  670,  822,  974]],

                           [[  86,  302,  518,  734,  950, 1166, 1382],
                            [ 110,  390,  670,  950, 1230, 1510, 1790],
                            [ 134,  478,  822, 1166, 1510, 1854, 2198]]])

    res_0 = mode_n_product(tensor=array_3d, matrix=A, mode=0)
    res_1 = mode_n_product(tensor=array_3d, matrix=B, mode=1)
    res_2 = mode_n_product(tensor=array_3d, matrix=C, mode=2)

    np.testing.assert_array_equal(true_res_0, res_0)
    np.testing.assert_array_equal(true_res_1, res_1)
    np.testing.assert_array_equal(true_res_2, res_2)

def test_unfold():
    """Tests for unfold"""

    I, J, K = 2, 3, 4
    array_3d = np.arange(I * J * K).reshape((I, J, K))

    true_res_0 = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    true_res_1 = np.array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                           [ 4,  5,  6,  7, 16, 17, 18, 19],
                           [ 8,  9, 10, 11, 20, 21, 22, 23]])

    true_res_2 = np.array([[ 0,  4,  8, 12, 16, 20],
                           [ 1,  5,  9, 13, 17, 21],
                           [ 2,  6, 10, 14, 18, 22],
                           [ 3,  7, 11, 15, 19, 23]])

    res_0 = unfold(tensor=array_3d, mode=0)
    res_1 = unfold(tensor=array_3d, mode=1)
    res_2 = unfold(tensor=array_3d, mode=2)

    np.testing.assert_array_equal(true_res_0, res_0)
    np.testing.assert_array_equal(true_res_1, res_1)
    np.testing.assert_array_equal(true_res_2, res_2)

def test_fold():
    array_0 = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    array_1 = np.array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                        [ 4,  5,  6,  7, 16, 17, 18, 19],
                        [ 8,  9, 10, 11, 20, 21, 22, 23]])

    array_2 = np.array([[ 0,  4,  8, 12, 16, 20],
                        [ 1,  5,  9, 13, 17, 21],
                        [ 2,  6, 10, 14, 18, 22],
                        [ 3,  7, 11, 15, 19, 23]])

    true_shape = (2,3,4)
    n_elements = reduce(lambda x, y: x*y, true_shape)
    true_res = np.arange(n_elements).reshape(true_shape)

    res_0 = fold(array_0, 0, true_shape)
    res_1 = fold(array_1, 1, true_shape)
    res_2 = fold(array_2, 2, true_shape)

    assert (res_0.shape == true_shape)
    assert (res_1.shape == true_shape)
    assert (res_2.shape == true_shape)

    np.testing.assert_array_equal(true_res, res_0)
    np.testing.assert_array_equal(true_res, res_1)
    np.testing.assert_array_equal(true_res, res_2)

