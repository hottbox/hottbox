"""
Tests for the tucker module of decomposition algorithms
"""
import pytest
import sys
import io
import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
from ..tucker import *
from ....core.structures import Tensor, TensorTKD
from ....pdtools import pd_to_tensor


class TestBaseTucker:
    """ Tests for BaseTucker class """
    def test_init(self):
        """ Tests for constructor of BaseTucker class """
        default_params = dict(process=(),
                              verbose=False
                              )

        # basically for coverage tests object of
        with pytest.raises(NotImplementedError):
            tensor = Tensor(np.arange(2))
            rank = 5
            keep_meta = 0
            base_tucker = BaseTucker(**default_params)
            base_tucker.decompose(tensor, rank, keep_meta)

        with pytest.raises(NotImplementedError):
            tensor = Tensor(np.arange(2))
            rank = 5
            base_tucker = BaseTucker(**default_params)
            base_tucker._init_fmat(tensor, rank)

        with pytest.raises(NotImplementedError):
            base_tucker = BaseTucker(**default_params)
            base_tucker.converged()

        with pytest.raises(NotImplementedError):
            base_tucker = BaseTucker(**default_params)
            base_tucker.plot()


class TestHOSVD:
    """ Tests for HOSVD class """
    def test_init(self):
        """ Tests for the constructor of HOSVD algorithm """
        process = (3, 1, 2)
        verbose = False
        hosvd = HOSVD(process=process,
                      verbose=verbose)

        assert hosvd.name == HOSVD.__name__
        assert hosvd.process == process
        assert hosvd.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        hosvd = HOSVD()
        hosvd.process = (3, 2, 1)
        hosvd_copy = hosvd.copy()

        assert hosvd_copy is not hosvd
        assert hosvd_copy.name == hosvd.name
        assert hosvd_copy.process == hosvd.process
        assert hosvd_copy.verbose == hosvd.verbose

        hosvd.process = (1, 2, 3)
        hosvd.scription = 'qwerty'
        hosvd.verbose = not hosvd.verbose
        assert hosvd_copy.process != hosvd.process
        assert hosvd_copy.verbose != hosvd.verbose

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        hosvd = HOSVD()
        tensor = Tensor(np.arange(5))
        rank = (1,)
        hosvd._init_fmat(tensor=tensor, rank=rank)
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_decompose(self):
        """ Tests for decompose method """
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        np.random.seed(0)

        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_3d)
        rank = (5, 6, 7)
        hosvd = HOSVD(verbose=True)

        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_tkd, TensorTKD)
        assert tensor_tkd.order == tensor.order
        assert tensor_tkd.rank == rank
        # check dimensionality of computed factor matrices
        for mode, fmat in enumerate(tensor_tkd.fmat):
            assert fmat.shape == (tensor.shape[mode], rank[mode])
        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for skipping modes if they are not specified in hosvd.process
        process_order = (1, 2)
        hosvd.process = process_order
        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank)
        for mode, fmat in enumerate(tensor_tkd.fmat):
            if mode in process_order:
                assert fmat.shape == (tensor.shape[mode], rank[mode])
            else:
                np.testing.assert_array_equal(fmat, np.eye(tensor.shape[mode]))

        # ------ tests perfect reconstruction
        rank = tensor.shape
        hosvd.process = ()
        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank)
        np.testing.assert_almost_equal(tensor_tkd.reconstruct().data, tensor.data)

        # ------ tests that should FAIL due to wrong input type
        hosvd = HOSVD()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2, 2, 2)
            hosvd.decompose(tensor=incorrect_tensor, rank=correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2, 2, 2]
            hosvd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2, 2)
            hosvd.decompose(tensor=correct_tensor, rank=incorrect_rank)

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
        rank = (2, 2, 2)
        hosvd = HOSVD()

        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank, keep_meta=2)
        assert tensor.modes == tensor_tkd.modes

        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank, keep_meta=1)
        assert all([tensor.modes[i].name == tensor_tkd.modes[i].name for i in range(tensor_tkd.order)])
        assert all([tensor_tkd.modes[i].index is None for i in range(tensor_tkd.order)])

        tensor_tkd = hosvd.decompose(tensor=tensor, rank=rank, keep_meta=0)
        tensor.reset_meta()
        assert tensor_tkd.modes == tensor.modes

    def test_converged(self):
        """ Tests for converged method """
        with pytest.warns(RuntimeWarning):
            hosvd = HOSVD()
            _ = hosvd.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        hosvd = HOSVD()
        hosvd.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed


class TestHOOI:
    """ Tests for HOOI class """
    def test_init(self):
        """ Tests for the constructor of HOSVD algorithm """
        init = 'hosvd'
        max_iter = 50
        epsilon = 10e-3
        tol = 10e-5
        process = (3, 2, 1)
        verbose = False
        hooi = HOOI(init=init,
                    max_iter=max_iter,
                    epsilon=epsilon,
                    tol=tol,
                    process=process,
                    verbose=verbose)
        assert not hooi.cost  # check that this list is empty
        assert hooi.name == HOOI.__name__
        assert hooi.init == init
        assert hooi.max_iter == max_iter
        assert hooi.epsilon == epsilon
        assert hooi.tol == tol
        assert hooi.process == process
        assert hooi.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        hooi = HOOI()
        hooi.process = (3, 2, 1)
        hooi.cost = [1, 2]
        hooi_copy = hooi.copy()

        assert hooi_copy is not hooi
        assert hooi_copy.name == hooi.name
        assert hooi_copy.init == hooi.init
        assert hooi_copy.max_iter == hooi.max_iter
        assert hooi_copy.epsilon == hooi.epsilon
        assert hooi_copy.tol == hooi.tol
        assert hooi_copy.process == hooi.process
        assert hooi_copy.verbose == hooi.verbose
        assert hooi_copy.cost != hooi.cost

        hooi.init = 'qwerty'
        hooi.max_iter += 1
        hooi.epsilon += 1
        hooi.tol += 1
        hooi.process = (1, 2, 3)
        hooi.verbose = not hooi.verbose
        hooi.cost = [3, 4]
        assert hooi_copy.init != hooi.init
        assert hooi_copy.max_iter != hooi.max_iter
        assert hooi_copy.epsilon != hooi.epsilon
        assert hooi_copy.tol != hooi.tol
        assert hooi_copy.process != hooi.process
        assert hooi_copy.verbose != hooi.verbose
        assert hooi.cost != hooi_copy.cost

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        np.random.seed(0)
        shape = (3, 4, 5)
        size = reduce(lambda x, y: x * y, shape)
        tensor = Tensor(np.random.randn(size).reshape(shape))
        hooi = HOOI()

        # ------ tests that hooi.cost is reset each time _init_fmat is called
        hooi.cost = [1, 2, 3]
        rank = tensor.shape
        hooi._init_fmat(tensor=tensor, rank=rank)
        assert not hooi.cost

        # ------ tests for the type of initialisation
        # hosvd type initialisation should produce factor matrices with orthogonal columns
        rank = (2, 3, 4)
        hooi = HOOI(init='hosvd')
        fmat = hooi._init_fmat(tensor=tensor, rank=rank)
        for mode, mat in enumerate(fmat):
            result = np.dot(mat.T, mat)
            true_result = np.eye(rank[mode])
            np.testing.assert_almost_equal(result, true_result)

        # unknown type of initialisation
        with pytest.raises(NotImplementedError):
            hooi = HOOI(init='qwerty')
            hooi._init_fmat(tensor=tensor, rank=rank)

    def test_decompose(self):
        """ Tests for decompose method """
        # ------ tests for termination conditions
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        np.random.seed(0)
        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_3d)
        rank = (2, 3, 4)
        hooi = HOOI(verbose=True)

        # check for termination at max iter
        hooi.max_iter = 5
        hooi.epsilon = 0.01
        hooi.tol = 0.0001
        hooi.decompose(tensor=tensor, rank=rank)
        assert not hooi.converged
        assert len(hooi.cost) == hooi.max_iter
        assert hooi.cost[-1] > hooi.epsilon

        # check for termination when acceptable level of approximation is achieved
        hooi.max_iter = 20
        hooi.epsilon = 0.84
        hooi.tol = 0.0001
        hooi.decompose(tensor=tensor, rank=rank)
        assert not hooi.converged
        assert len(hooi.cost) < hooi.max_iter
        assert hooi.cost[-1] <= hooi.epsilon

        # check for termination at convergence
        hooi.max_iter = 20
        hooi.epsilon = 0.01
        hooi.tol = 0.0001
        hooi.decompose(tensor=tensor, rank=rank)
        assert hooi.converged
        assert len(hooi.cost) < hooi.max_iter
        assert hooi.cost[-1] > hooi.epsilon

        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for correct output type and values
        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_3d)
        rank = (2, 3, 4)

        hooi = HOOI(init='hosvd', max_iter=50, epsilon=10e-3, tol=10e-5)

        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_tkd, TensorTKD)
        assert tensor_tkd.order == tensor.order
        assert tensor_tkd.rank == rank
        # check dimensionality of computed factor matrices
        for mode, fmat in enumerate(tensor_tkd.fmat):
            assert fmat.shape == (tensor.shape[mode], rank[mode])

        # ------ tests for skipping modes if they are not specified in hooi.process
        process_order = (1, 2)
        hooi.process = process_order
        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank)
        for mode, fmat in enumerate(tensor_tkd.fmat):
            if mode in process_order:
                assert fmat.shape == (tensor.shape[mode], rank[mode])
            else:
                np.testing.assert_array_equal(fmat, np.eye(tensor.shape[mode]))

        # ------ tests perfect reconstruction
        rank = tensor.shape
        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank)
        tensor_rec = tensor_tkd.reconstruct()
        np.testing.assert_almost_equal(tensor_rec.data, tensor.data)

        # ------ tests that should FAIL due to wrong input type
        hooi = HOOI()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2, 2, 2)
            hooi.decompose(tensor=incorrect_tensor, rank=correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2, 2, 2]
            hooi.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2, 2)
            hooi.decompose(tensor=correct_tensor, rank=incorrect_rank)

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
        rank = (2, 2, 2)
        hooi = HOOI()

        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank, keep_meta=2)
        assert tensor_tkd.modes == tensor.modes

        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank, keep_meta=1)
        assert all([tensor_tkd.modes[i].name == tensor.modes[i].name for i in range(tensor_tkd.order)])
        assert all([tensor_tkd.modes[i].index is None for i in range(tensor_tkd.order)])

        tensor_tkd = hooi.decompose(tensor=tensor, rank=rank, keep_meta=0)
        tensor.reset_meta()
        assert tensor_tkd.modes == tensor.modes

    def test_converged(self):
        """ Tests for converged method """
        tol = 0.01
        hooi = HOOI(tol=tol)

        # when it is empty, which is the case at the object creation
        assert not hooi.converged

        # requires at least two values
        hooi.cost = [0.001]
        assert not hooi.converged

        # difference greater then `tol`
        hooi.cost = [0.1, 0.2]
        assert not hooi.converged

        # checks only the last two values
        hooi.cost = [0.0001, 0.0002, 0.1, 0.2]
        assert not hooi.converged

        hooi.cost = [0.001, 0.0001]
        assert hooi.converged

        hooi.cost = [0.1, 0.2, 0.001, 0.0001]
        assert hooi.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()  # Create StringIO object for testing verbosity
        sys.stdout = captured_output  # and redirect stdout.
        hooi = HOOI()
        hooi.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed
