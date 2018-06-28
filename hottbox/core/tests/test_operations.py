"""
Tests for the operations module
"""
import pytest
import numpy as np
from functools import reduce
from ..operations import *


def test_mode_n_product():
    """ Tests for the `mode_n_product` function """
    I, J, K = 2, 3, 4
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

    # matrix should be a 2-D array
    with pytest.raises(ValueError):
        incorrect_matrix = np.arange(I_new)
        mode_n_product(tensor=array_3d, matrix=incorrect_matrix, mode=0)


def test_unfold():
    """ Tests for `unfold` function """
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
    assert res_0 is not array_3d  # check that not references
    assert res_1 is not array_3d  # check that not references
    assert res_2 is not array_3d  # check that not references


def test_kolda_unfold():
    """ Tests for `kolda_unfold` function """
    I, J, K = 2, 3, 4
    array_3d = np.arange(I * J * K).reshape((I, J, K))

    true_res_0 = np.array([[ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11],
                           [12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]])

    true_res_1 = np.array([[ 0, 12,  1, 13,  2, 14,  3, 15],
                           [ 4, 16,  5, 17,  6, 18,  7, 19],
                           [ 8, 20,  9, 21, 10, 22, 11, 23]])

    true_res_2 = np.array([[ 0, 12,  4, 16,  8, 20],
                           [ 1, 13,  5, 17,  9, 21],
                           [ 2, 14,  6, 18, 10, 22],
                           [ 3, 15,  7, 19, 11, 23]])

    res_0 = kolda_unfold(tensor=array_3d, mode=0)
    res_1 = kolda_unfold(tensor=array_3d, mode=1)
    res_2 = kolda_unfold(tensor=array_3d, mode=2)

    np.testing.assert_array_equal(true_res_0, res_0)
    np.testing.assert_array_equal(true_res_1, res_1)
    np.testing.assert_array_equal(true_res_2, res_2)
    assert res_0 is not array_3d  # check that not references
    assert res_1 is not array_3d  # check that not references
    assert res_2 is not array_3d  # check that not references


def test_fold():
    """ Test for `fold` function """
    array_0 = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    array_1 = np.array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                        [ 4,  5,  6,  7, 16, 17, 18, 19],
                        [ 8,  9, 10, 11, 20, 21, 22, 23]])

    array_2 = np.array([[ 0,  4,  8, 12, 16, 20],
                        [ 1,  5,  9, 13, 17, 21],
                        [ 2,  6, 10, 14, 18, 22],
                        [ 3,  7, 11, 15, 19, 23]])

    true_shape = (2, 3, 4)
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

    assert res_0 is not array_0  # check that not references
    assert res_1 is not array_1  # check that not references
    assert res_2 is not array_2  # check that not references


def test_kolda_fold():
    """ Test for `kolda_fold` function """
    array_0 = np.array([[ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11],
                        [12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]])

    array_1 = np.array([[ 0, 12,  1, 13,  2, 14,  3, 15],
                        [ 4, 16,  5, 17,  6, 18,  7, 19],
                        [ 8, 20,  9, 21, 10, 22, 11, 23]])

    array_2 = np.array([[ 0, 12,  4, 16,  8, 20],
                        [ 1, 13,  5, 17,  9, 21],
                        [ 2, 14,  6, 18, 10, 22],
                        [ 3, 15,  7, 19, 11, 23]])

    true_shape = (2, 3, 4)
    n_elements = reduce(lambda x, y: x * y, true_shape)
    true_res = np.arange(n_elements).reshape(true_shape)

    res_0 = kolda_fold(array_0, 0, true_shape)
    res_1 = kolda_fold(array_1, 1, true_shape)
    res_2 = kolda_fold(array_2, 2, true_shape)

    assert (res_0.shape == true_shape)
    assert (res_1.shape == true_shape)
    assert (res_2.shape == true_shape)

    np.testing.assert_array_equal(true_res, res_0)
    np.testing.assert_array_equal(true_res, res_1)
    np.testing.assert_array_equal(true_res, res_2)

    assert res_0 is not array_0  # check that not references
    assert res_1 is not array_1  # check that not references
    assert res_2 is not array_2  # check that not references


def test_khatri_rao():
    """ Tests for `khatri_rao` function """
    np.random.seed(0)
    n_col = 4
    rows = [2, 3, 4]
    n_rows = reduce(lambda x, y: x * y, rows)
    true_shape = (n_rows, n_col)
    matrices = [np.random.randn(n_row, n_col) for n_row in rows]

    result = khatri_rao(matrices=matrices)
    assert result.shape == true_shape

    result = khatri_rao(matrices=matrices, reverse=True)
    assert result.shape == true_shape

    result = khatri_rao(matrices=matrices, skip_matrix=0)
    n_rows = reduce(lambda x, y: x * y, rows[1:])
    true_shape = (n_rows, n_col)
    assert result.shape == true_shape

    result = khatri_rao(matrices=matrices, skip_matrix=(len(rows)-1))
    n_rows = reduce(lambda x, y: x * y, rows[:-1])
    true_shape = (n_rows, n_col)
    assert result.shape == true_shape

    # ------ tests that should FAIL
    # Require a list of at least of two matrices
    with pytest.raises(ValueError):
        matrix = np.random.randn(2, 3)
        wrong_matrices = [matrix]
        khatri_rao(matrices=wrong_matrices)

    # All matrices should have the same number of columns
    with pytest.raises(ValueError):
        n_cols = 3
        wrong_matrices = [np.random.randn(n_rows, n_cols) for n_rows in range(2, 5)]
        wrong_matrices.append(np.random.randn(2, n_cols+1))
        khatri_rao(matrices=wrong_matrices)


def test_hadamard():
    """ Tests for `hadamard` function """
    np.random.seed(0)
    n_col = 2
    n_row = 33
    true_shape = (n_row, n_col)
    matrices = [np.random.randn(n_row, n_col) for _ in range(3)]

    result = hadamard(matrices=matrices, reverse=False)
    assert result.shape == true_shape

    result = hadamard(matrices=matrices, reverse=True)
    assert result.shape == true_shape

    result = hadamard(matrices=matrices, skip_matrix=0)
    assert result.shape == true_shape

    result = hadamard(matrices=matrices, skip_matrix=(len(matrices) - 1))
    assert result.shape == true_shape


def test_kronecker():
    """ Tests for `kronecker` function """
    np.random.seed(0)
    rows = [2, 3, 4]
    cols = [5, 6, 7]
    n_rows = reduce(lambda x, y: x * y, rows)
    n_cols = reduce(lambda x, y: x * y, cols)
    true_shape = (n_rows, n_cols)
    matrices = [np.random.randn(rows[i], cols[i]) for i in range(len(rows))]

    result = kronecker(matrices=matrices)
    assert result.shape == true_shape

    result = kronecker(matrices=matrices, reverse=True)
    assert result.shape == true_shape

    result = kronecker(matrices=matrices, skip_matrix=0)
    n_rows = reduce(lambda x, y: x * y, rows[1:])
    n_cols = reduce(lambda x, y: x * y, cols[1:])
    true_shape = (n_rows, n_cols)
    assert result.shape == true_shape

    result = kronecker(matrices=matrices, skip_matrix=(len(rows) - 1))
    n_rows = reduce(lambda x, y: x * y, rows[:-1])
    n_cols = reduce(lambda x, y: x * y, cols[:-1])
    true_shape = (n_rows, n_cols)
    assert result.shape == true_shape
