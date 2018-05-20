"""
Tests for the tensor train module of decomposition algorithms
"""
import pytest
import sys
import io
import pandas as pd
import numpy as np
from functools import reduce
from itertools import product
from ..tensor_train import *
from ....core.structures import Tensor, TensorTT
from ....pdtools import pd_to_tensor


class TestBaseTensorTrain:
    """ Tests for BaseTensorTrain class """
    def test_init(self):
        """ Tests for constructor of BaseTensorTrain class """
        default_params = dict(verbose=False)

        # basically for coverage tests object of
        with pytest.raises(NotImplementedError):
            tensor = Tensor(np.arange(2))
            rank = 5
            base_tucker = BaseTensorTrain(**default_params)
            base_tucker.decompose(tensor, rank)

        with pytest.raises(NotImplementedError):
            tensor = Tensor(np.arange(2))
            rank = 5
            base_tucker = BaseTensorTrain(**default_params)
            base_tucker._init_fmat(tensor, rank)

        with pytest.raises(NotImplementedError):
            base_tucker = BaseTensorTrain(**default_params)
            base_tucker.converged()

        with pytest.raises(NotImplementedError):
            base_tucker = BaseTensorTrain(**default_params)
            base_tucker.plot()


class TestTTSVD:
    """ Tests for TTSVD class """
    def test_init(self):
        """ Tests for the constructor of HOSVD algorithm """
        verbose = False
        ttsvd = TTSVD(verbose=verbose)

        assert ttsvd.name == TTSVD.__name__
        assert ttsvd.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        ttsvd = TTSVD()
        ttsvd_copy = ttsvd.copy()

        assert ttsvd_copy is not ttsvd
        assert ttsvd_copy.name == ttsvd.name
        assert ttsvd_copy.verbose == ttsvd.verbose

        ttsvd.process = (1, 2, 3)
        ttsvd.verbose = not ttsvd.verbose
        assert ttsvd_copy.verbose != ttsvd.verbose

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        ttsvd = TTSVD()
        tensor = Tensor(np.arange(5))
        rank = (1,)
        ttsvd._init_fmat(tensor=tensor, rank=rank)
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_decompose(self):
        """ Tests for decompose method """
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        np.random.seed(0)
        ttsvd = TTSVD(verbose=False)

        # ------ tests for 3-D data
        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_data = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_data)
        rank = (2, 3)
        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_tt, TensorTT)
        assert tensor_tt.order == tensor.order
        assert tensor_tt.rank == rank
        # check dimensionality of computed cores
        assert tensor_tt.core(0).shape == (shape[0], rank[0])
        assert tensor_tt.core(1).shape == (rank[0], shape[1], rank[1])
        assert tensor_tt.core(2).shape == (rank[1], shape[2])
        assert captured_output.getvalue() == ''  # to check that nothing was actually printed

        # ------ tests for 4-D data
        shape = (5, 6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_data = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_data)
        rank = (2, 3, 4)
        ttsvd.verbose = True
        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_tt, TensorTT)
        assert tensor_tt.order == tensor.order
        assert tensor_tt.rank == rank
        # check dimensionality of computed cores
        assert tensor_tt.core(0).shape == (shape[0], rank[0])
        assert tensor_tt.core(1).shape == (rank[0], shape[1], rank[1])
        assert tensor_tt.core(2).shape == (rank[1], shape[2], rank[2])
        assert tensor_tt.core(3).shape == (rank[2], shape[3])
        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for 4-D data
        shape = (3, 4, 5, 6)
        size = reduce(lambda x, y: x * y, shape)
        array_data = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_data)
        rank = (2, 3, 4)
        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_tt, TensorTT)
        assert tensor_tt.order == tensor.order
        assert tensor_tt.rank == rank
        # check dimensionality of computed cores
        assert tensor_tt.core(0).shape == (shape[0], rank[0])
        assert tensor_tt.core(1).shape == (rank[0], shape[1], rank[1])
        assert tensor_tt.core(2).shape == (rank[1], shape[2], rank[2])
        assert tensor_tt.core(3).shape == (rank[2], shape[3])

        # ------ tests that should FAIL due to wrong input type
        ttsvd = TTSVD()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2, 2)
            ttsvd.decompose(tensor=incorrect_tensor, rank=correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2, 2]
            ttsvd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = tuple([2 for _ in range(correct_tensor.order)])
            ttsvd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect values of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [correct_tensor.shape[mode] for mode in range(correct_tensor.order-1)]
            incorrect_rank[0] += 1
            ttsvd.decompose(tensor=correct_tensor, rank=tuple(incorrect_rank))
        # check for the last value of rank to be incorrect (greater then the last dim size of a tensor)
        with pytest.raises(ValueError):
            shape = (3, 3, 3, 3, 3, 2)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [correct_tensor.shape[mode] for mode in range(correct_tensor.order-1)]
            ttsvd.decompose(tensor=correct_tensor, rank=tuple(incorrect_rank))

    def test_decompose_with_meta(self):
        """ Tests for keeping meta data about modes """
        content = dict(
            country=['UK', 'RUS'],
            year=[2005, 2015, 2010],
            month=['Jan', 'Feb', 'Mar', 'Apr']
        )
        data = list(product(*content.values()))
        columns = list(content.keys())
        df = pd.DataFrame(data=data, columns=columns)
        df['population'] = np.arange(df.shape[0], dtype='float32')
        df_mi = df.set_index(columns)
        tensor = pd_to_tensor(df=df_mi, keep_index=True)
        rank = (2,2)
        ttsvd = TTSVD()

        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank, keep_meta=2)
        assert tensor_tt.modes == tensor.modes

        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank, keep_meta=1)
        assert all([tensor_tt.modes[i].name == tensor.modes[i].name for i in range(tensor_tt.order)])
        assert all([tensor_tt.modes[i].index is None for i in range(tensor_tt.order)])

        tensor_tt = ttsvd.decompose(tensor=tensor, rank=rank, keep_meta=0)
        tensor.reset_meta()
        assert tensor_tt.modes == tensor.modes

    def test_converged(self):
        """ Tests for converged method """
        with pytest.warns(RuntimeWarning):
            ttsvd = TTSVD()
            _ = ttsvd.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        ttsvd = TTSVD()
        ttsvd.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed


def test_svd_tt():
    """ Tests for _svd_tt """
    from ..tensor_train import _svd_tt
    rank = 3
    matrix = np.random.randn(6, 4)
    result = _svd_tt(matrix=matrix, rank=rank)
    assert result[0].shape == (matrix.shape[0], rank)
    assert result[1].shape == (rank,)
    assert result[2].shape == (matrix.shape[1], rank)
