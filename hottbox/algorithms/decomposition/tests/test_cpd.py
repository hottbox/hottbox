"""
Tests for the cpd module
"""
import pytest
import sys
import io
import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
from hottbox.core.structures import Tensor, TensorCPD
from hottbox.pdtools import pd_to_tensor
from ..cpd import BaseCPD, CPD, RandomisedCPD, Parafac2


class TestBaseCPD:
    """ Tests for BaseCPD class """

    def test_init(self):
        """ Tests for constructor of BaseCPD class """
        # Even though we can create such object we shouldn't do that
        default_params = dict(init='svd',
                              max_iter=50,
                              epsilon=10e-3,
                              tol=10e-5,
                              random_state=None,
                              verbose=False
                              )

        # basically for coverage tests object of
        with pytest.raises(NotImplementedError):
            tensor = Tensor(np.arange(2))
            rank = 5
            keep_meta = 0
            base_cpd = BaseCPD(**default_params)
            base_cpd.decompose(tensor, rank, keep_meta)
        with pytest.raises(NotImplementedError):
            base_cpd = BaseCPD(**default_params)
            base_cpd.plot()


class TestCPD:
    """ Tests for CPD class """

    def test_init(self):
        """ Tests for the constructor of CPD algorithm """
        init = 'svd'
        max_iter = 50
        epsilon = 10e-3
        tol = 10e-5
        verbose = False
        cpd = CPD(init=init,
                  max_iter=max_iter,
                  epsilon=epsilon,
                  tol=tol,
                  verbose=verbose)
        assert not cpd.cost         # check that this list is empty
        assert cpd.name == CPD.__name__
        assert cpd.init == init
        assert cpd.max_iter == max_iter
        assert cpd.epsilon == epsilon
        assert cpd.tol == tol
        assert cpd.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        cpd = CPD()
        cpd.cost = [1, 2]
        cpd_copy = cpd.copy()

        assert cpd_copy is not cpd
        assert cpd_copy.name == cpd.name
        assert cpd_copy.init == cpd.init
        assert cpd_copy.max_iter == cpd.max_iter
        assert cpd_copy.epsilon == cpd.epsilon
        assert cpd_copy.tol == cpd.tol
        assert cpd_copy.verbose == cpd.verbose
        assert cpd_copy.cost != cpd.cost

        cpd.init = 'qwerty'
        cpd.max_iter += 1
        cpd.epsilon += 1
        cpd.tol += 1
        cpd.verbose = not cpd.verbose
        cpd.cost = [3, 4]
        assert cpd_copy.init != cpd.init
        assert cpd_copy.max_iter != cpd.max_iter
        assert cpd_copy.epsilon != cpd.epsilon
        assert cpd_copy.tol != cpd.tol
        assert cpd_copy.verbose != cpd.verbose
        assert cpd.cost != cpd_copy.cost

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        np.random.seed(0)
        shape = (4, 5, 6)
        size = reduce(lambda x, y: x * y, shape)
        tensor = Tensor(np.random.randn(size).reshape(shape))
        cpd = CPD()

        # ------ tests on getting factor matrices of the correct shape
        for rank_value in range(min(tensor.shape)-1, max(tensor.shape)+2):
            rank = (rank_value,)
            fmat = cpd._init_fmat(tensor=tensor, rank=rank)
            for mode, mat in enumerate(fmat):
                assert mat.shape == (tensor.shape[mode], rank_value)

        # ------ tests for the type of initialisation
        # svd type initialisation should produce factor matrices with orthogonal columns
        rank = (min(tensor.shape)-1,)
        cpd = CPD(init='svd')
        fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result = np.dot(mat.T, mat)
            true_result = np.eye(rank[0])
            np.testing.assert_almost_equal(result, true_result)

        # svd type initialisation but the `rank` is greater then one of the dimensions then you get random fmat
        # and there would be a runtime warning
        rank = (min(tensor.shape)+1,)
        cpd = CPD(init='svd', verbose=True)
        with pytest.warns(RuntimeWarning):
            fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result_1 = np.dot(mat.T, mat)
            result_2 = np.eye(rank[0])
            # since each mat is randomly initialized it is not orthonormal
            with pytest.raises(AssertionError):
                np.testing.assert_almost_equal(result_1, result_2)

        # random type initialisation should produce factor matrices each of which is not orthonormal
        rank = (3,)
        cpd = CPD(init='random')
        fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result_1 = np.dot(mat.T, mat)
            result_2 = np.eye(rank[0])
            # since each mat is randomly initialized it is not orthonormal
            with pytest.raises(AssertionError):
                np.testing.assert_almost_equal(result_1, result_2)

        # unknown type of initialisation
        with pytest.raises(NotImplementedError):
            rank = (min(tensor.shape)-1,)
            cpd = CPD(init='qwerty')
            cpd._init_fmat(tensor=tensor, rank=rank)

    def test_decompose(self):
        """ Tests for decompose method """
        # ------ tests for termination conditions
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        np.random.seed(0)
        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_3d)
        rank = (2,)
        cpd = CPD(verbose=True)

        # check for termination at max iter
        cpd.max_iter = 10
        cpd.epsilon = 0.01
        cpd.tol = 0.0001
        cpd.decompose(tensor=tensor, rank=rank)
        assert not cpd.converged
        assert len(cpd.cost) == cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon
        
        # Repeat cpd, test is self.cost is reset
        cpd.decompose(tensor=tensor, rank=rank)
        assert len(cpd.cost) == cpd.max_iter
        
        # check for termination when acceptable level of approximation is achieved
        cpd.max_iter = 20
        cpd.epsilon = 0.91492
        cpd.tol = 0.0001
        cpd.decompose(tensor=tensor, rank=rank)
        assert not cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] <= cpd.epsilon

        # check for termination at convergence
        cpd.max_iter = 20
        cpd.epsilon = 0.01
        cpd.tol = 0.0001
        cpd.decompose(tensor=tensor, rank=rank)
        assert cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon

        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for correct output type and values

        shape = (4, 5, 6)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.arange(size, dtype='float32').reshape(shape)
        tensor = Tensor(array_3d)
        rank = (7,)

        cpd = CPD(init='random', max_iter=50, epsilon=10e-3, tol=10e-5)

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_cpd, TensorCPD)
        assert tensor_cpd.order == tensor.order
        assert tensor_cpd.rank == rank
        # check dimensionality of computed factor matrices
        for mode, fmat in enumerate(tensor_cpd.fmat):
            assert fmat.shape == (tensor.shape[mode], rank[0])

        tensor_rec = tensor_cpd.reconstruct()
        np.testing.assert_almost_equal(tensor_rec.data, tensor.data)

        # ------ tests that should FAIL due to wrong input type
        cpd = CPD()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2,)
            cpd.decompose(tensor=incorrect_tensor, rank=correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2]
            cpd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2, 3)
            cpd.decompose(tensor=correct_tensor, rank=incorrect_rank)

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
        rank = (2,)
        cpd = CPD()

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=2)
        assert tensor_cpd.modes == tensor.modes

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=1)
        assert all([tensor_cpd.modes[i].name == tensor.modes[i].name for i in range(tensor_cpd.order)])
        assert all([tensor_cpd.modes[i].index is None for i in range(tensor_cpd.order)])

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=0)
        tensor.reset_meta()
        assert tensor_cpd.modes == tensor.modes

    def test_converged(self):
        """ Tests for converged method """
        tol = 0.01
        cpd = CPD(tol=tol)

        # when it is empty, which is the case at the object creation
        assert not cpd.converged

        # requires at least two values
        cpd.cost = [0.001]
        assert not cpd.converged

        # difference greater then `tol`
        cpd.cost = [0.1, 0.2]
        assert not cpd.converged

        # checks only the last two values
        cpd.cost = [0.0001, 0.0002, 0.1, 0.2]
        assert not cpd.converged

        cpd.cost = [0.001, 0.0001]
        assert cpd.converged

        cpd.cost = [0.1, 0.2, 0.001, 0.0001]
        assert cpd.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        cpd = CPD()
        cpd.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed


class TestRandomisedCPD:
    """ Tests for CpRand class """

    def test_init(self):
        """ Tests for the constructor of CPD algorithm """
        init = 'svd'
        max_iter = 50
        epsilon = 10e-3
        tol = 10e-5
        verbose = False
        sample_size = None
        cpd = RandomisedCPD(init=init,
                            max_iter=max_iter,
                            epsilon=epsilon,
                            tol=tol,
                            verbose=verbose,
                            sample_size=sample_size)
        assert not cpd.cost         # check that this list is empty
        assert cpd.name == RandomisedCPD.__name__
        assert cpd.init == init
        assert cpd.max_iter == max_iter
        assert cpd.epsilon == epsilon
        assert cpd.tol == tol
        assert cpd.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        cpd = RandomisedCPD()
        cpd.cost = [1, 2]
        cpd_copy = cpd.copy()

        assert cpd_copy is not cpd
        assert cpd_copy.name == cpd.name
        assert cpd_copy.init == cpd.init
        assert cpd_copy.max_iter == cpd.max_iter
        assert cpd_copy.epsilon == cpd.epsilon
        assert cpd_copy.tol == cpd.tol
        assert cpd_copy.verbose == cpd.verbose
        assert cpd_copy.cost != cpd.cost

        cpd.init = 'qwerty'
        cpd.max_iter += 1
        cpd.epsilon += 1
        cpd.tol += 1
        cpd.verbose = not cpd.verbose
        cpd.cost = [3, 4]
        assert cpd_copy.init != cpd.init
        assert cpd_copy.max_iter != cpd.max_iter
        assert cpd_copy.epsilon != cpd.epsilon
        assert cpd_copy.tol != cpd.tol
        assert cpd_copy.verbose != cpd.verbose
        assert cpd.cost != cpd_copy.cost

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        np.random.seed(0)
        shape = (4, 5, 6)
        size = reduce(lambda x, y: x * y, shape)
        tensor = Tensor(np.random.randn(size).reshape(shape))
        cpd = RandomisedCPD()

        # ------ tests on getting factor matrices of the correct shape
        for rank_value in range(min(tensor.shape)-1, max(tensor.shape)+2):
            rank = (rank_value,)
            fmat = cpd._init_fmat(tensor=tensor, rank=rank)
            for mode, mat in enumerate(fmat):
                assert mat.shape == (tensor.shape[mode], rank_value)

        # ------ tests for the type of initialisation
        # svd type initialisation should produce factor matrices with orthogonal columns
        rank = (min(tensor.shape)-1,)
        cpd = RandomisedCPD(init='svd')
        fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result = np.dot(mat.T, mat)
            true_result = np.eye(rank[0])
            np.testing.assert_almost_equal(result, true_result)

        # svd type initialisation but the `rank` is greater then one of the dimensions then you get random fmat
        # and there would be a runtime warning
        rank = (min(tensor.shape)+1,)
        cpd = RandomisedCPD(init='svd', verbose=True)
        with pytest.warns(RuntimeWarning):
            fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result_1 = np.dot(mat.T, mat)
            result_2 = np.eye(rank[0])
            # since each mat is randomly initialized it is not orthonormal
            with pytest.raises(AssertionError):
                np.testing.assert_almost_equal(result_1, result_2)

        # random type initialisation should produce factor matrices each of which is not orthonormal
        rank = (3,)
        cpd = RandomisedCPD(init='random')
        fmat = cpd._init_fmat(tensor=tensor, rank=rank)
        for mat in fmat:
            result_1 = np.dot(mat.T, mat)
            result_2 = np.eye(rank[0])
            # since each mat is randomly initialized it is not orthonormal
            with pytest.raises(AssertionError):
                np.testing.assert_almost_equal(result_1, result_2)

        # unknown type of initialisation
        with pytest.raises(NotImplementedError):
            rank = (min(tensor.shape)-1,)
            cpd = RandomisedCPD(init='qwerty')
            cpd._init_fmat(tensor=tensor, rank=rank)

    def test_decompose(self):
        """ Tests for decompose method """
        # ------ tests for termination conditions
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        np.random.seed(0)
        shape = (6, 7, 8)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.random.randn(size).reshape(shape)
        tensor = Tensor(array_3d)
        rank = (2,)
        cpd = RandomisedCPD(verbose=True)

        # check for termination at max iter
        cpd.max_iter = 10
        cpd.epsilon = 0.01
        cpd.tol = 0.0001
        cpd.decompose(tensor=tensor, rank=rank)
        assert not cpd.converged
        assert len(cpd.cost) == cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon
        
        # Repeat cpd, test is self.cost is reset
        cpd.decompose(tensor=tensor, rank=rank)
        assert len(cpd.cost) == cpd.max_iter
        
        # check for termination when acceptable level of approximation is achieved
        cpd.max_iter = 20
        cpd.epsilon = 0.98
        cpd.tol = 0.0001
        cpd.decompose(tensor=tensor, rank=rank)
        assert not cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] <= cpd.epsilon

        # check for termination at convergence
        cpd.max_iter = 20
        cpd.epsilon = 0.01
        cpd.tol = 0.03
        cpd.decompose(tensor=tensor, rank=rank)
        assert cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon

        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for correct output type and values

        shape = (4, 5, 6)
        size = reduce(lambda x, y: x * y, shape)
        array_3d = np.arange(size, dtype='float32').reshape(shape)
        tensor = Tensor(array_3d)
        rank = (7,)

        cpd = RandomisedCPD(init='random', max_iter=50, epsilon=10e-3, tol=10e-5)

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank)
        assert isinstance(tensor_cpd, TensorCPD)
        assert tensor_cpd.order == tensor.order
        assert tensor_cpd.rank == rank
        # check dimensionality of computed factor matrices
        for mode, fmat in enumerate(tensor_cpd.fmat):
            assert fmat.shape == (tensor.shape[mode], rank[0])

        tensor_rec = tensor_cpd.reconstruct()
        np.testing.assert_almost_equal(tensor_rec.data, tensor.data)

        # ------ tests that should FAIL due to wrong input type
        cpd = RandomisedCPD()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2,)
            cpd.decompose(tensor=incorrect_tensor, rank=correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2]
            cpd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2, 3)
            cpd.decompose(tensor=correct_tensor, rank=incorrect_rank)
        # invalid sample size
        with pytest.raises(ValueError):
            cpd = RandomisedCPD(sample_size=0)
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2,)
            cpd.decompose(tensor=correct_tensor, rank=incorrect_rank)

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
        rank = (2,)
        cpd = RandomisedCPD()

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=2)
        assert tensor_cpd.modes == tensor.modes

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=1)
        assert all([tensor_cpd.modes[i].name == tensor.modes[i].name for i in range(tensor_cpd.order)])
        assert all([tensor_cpd.modes[i].index is None for i in range(tensor_cpd.order)])

        tensor_cpd = cpd.decompose(tensor=tensor, rank=rank, keep_meta=0)
        tensor.reset_meta()
        assert tensor_cpd.modes == tensor.modes

    def test_converged(self):
        """ Tests for converged method """
        tol = 0.01
        cpd = RandomisedCPD(tol=tol)

        # when it is empty, which is the case at the object creation
        assert not cpd.converged

        # requires at least two values
        cpd.cost = [0.001]
        assert not cpd.converged

        # difference greater then `tol`
        cpd.cost = [0.1, 0.2]
        assert not cpd.converged

        # checks only the last two values
        cpd.cost = [0.0001, 0.0002, 0.1, 0.2]
        assert not cpd.converged

        cpd.cost = [0.001, 0.0001]
        assert cpd.converged

        cpd.cost = [0.1, 0.2, 0.001, 0.0001]
        assert cpd.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        cpd = RandomisedCPD()
        cpd.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed


class TestParafac2:
    """ Tests for CpRand class """

    def test_init(self):
        """ Tests for the constructor of CPD algorithm """
        max_iter = 50
        epsilon = 10e-3
        tol = 10e-5
        verbose = False
        sample_size = None
        cpd = Parafac2(max_iter=max_iter,
                       epsilon=epsilon,
                       tol=tol,
                       verbose=verbose)
        assert not cpd.cost         # check that this list is empty
        assert cpd.name == Parafac2.__name__
        assert cpd.max_iter == max_iter
        assert cpd.epsilon == epsilon
        assert cpd.tol == tol
        assert cpd.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        cpd = Parafac2()
        cpd.cost = [1, 2]
        cpd_copy = cpd.copy()

        assert cpd_copy is not cpd
        assert cpd_copy.name == cpd.name
        assert cpd_copy.init == cpd.init
        assert cpd_copy.max_iter == cpd.max_iter
        assert cpd_copy.epsilon == cpd.epsilon
        assert cpd_copy.tol == cpd.tol
        assert cpd_copy.verbose == cpd.verbose
        assert cpd_copy.cost != cpd.cost

        cpd.max_iter += 1
        cpd.epsilon += 1
        cpd.tol += 1
        cpd.verbose = not cpd.verbose
        cpd.cost = [3, 4]
        assert cpd_copy.max_iter != cpd.max_iter
        assert cpd_copy.epsilon != cpd.epsilon
        assert cpd_copy.tol != cpd.tol
        assert cpd_copy.verbose != cpd.verbose
        assert cpd.cost != cpd_copy.cost

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        np.random.seed(0)
        K = 5
        J = np.random.randint(15)
        I_k = np.random.randint(3,15,K)
        size = np.array([(_a,J) for _a in I_k])
        rank = (min(I_k + [J])-1,)
        tenL = [np.random.randn(*sz) for sz in size]
        cpd = Parafac2()

        # ------ correct shape and type for factor matrices
        # svd type initialisation should produce factor matrices with orthogonal columns
        r = rank[0]
        H, V, S, U = cpd._init_fmat(rank, size)
        assert H.shape == (r, r)
        assert V.shape == (J, r)
        assert S.shape == (r, r, K)
        for i, mat in enumerate(U):
            assert mat.shape == (I_k[i], r) 

        # ------ test for rank: expected to warn
        # Rank specified should be the match the specified shape
        rank = I_k[0]+1
        with pytest.warns(RuntimeWarning):
            cpd._init_fmat((rank,), size)

        #  ------ test for incorrect rank type 
        rank = I_k[0]
        with pytest.raises(IndexError):
            cpd._init_fmat(rank, size)

    def test_decompose(self):
        """ Tests for decompose method """
        # ------ tests for termination conditions
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        np.random.seed(0)
        K = 3
        J = 3
        I_k = (4,5,6)
        size = np.array([(_a,J) for _a in I_k])
        rank = (3,)
        tenL = [np.random.randn(*sz) for sz in size]
        cpd = Parafac2(verbose=True)

        # check for termination at max iter
        cpd.max_iter = 10
        cpd.epsilon = 0.01
        cpd.tol = 0.0001
        cpd.decompose(tenL, rank)
        assert not cpd.converged
        assert len(cpd.cost) == cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon

        # Repeat cpd, test is self.cost is reset
        cpd.decompose(tenL, rank)
        assert len(cpd.cost) == cpd.max_iter
        
        # check for termination when acceptable level of approximation is achieved
        cpd.max_iter = 10
        cpd.epsilon = 0.98
        cpd.tol = 10e-5
        cpd.decompose(tenL, rank)
        assert not cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] <= cpd.epsilon

        # check for termination at convergence
        cpd.max_iter = 20
        cpd.epsilon = 0.01
        cpd.tol = 0.1
        cpd.decompose(tenL, rank)
        assert cpd.converged
        assert len(cpd.cost) < cpd.max_iter
        assert cpd.cost[-1] > cpd.epsilon

        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for correct output type and values
        cpd = Parafac2(max_iter=100, epsilon=10e-3, tol=10e-5)

        U, S, V, tensor_rec = cpd.decompose(tenL, rank)
        # types
        assert isinstance(U, np.ndarray)
        assert isinstance(S, np.ndarray)
        assert isinstance(V, np.ndarray)
        # dimensions

        assert S.shape == (rank[0], rank[0], K)
        assert V.shape == (J, rank[0])
        # check dimensionality of computed factor matrices
        for i, mat in enumerate(U):
            assert mat.shape == (I_k[i], rank[0])

        for i in range(len(tenL)):
            np.testing.assert_almost_equal(tensor_rec[i], tenL[i], decimal=1)

        # ------ tests that should FAIL due to wrong input type
        cpd = Parafac2()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2,)
            cpd.decompose(incorrect_tensor, correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_rank = [2]
            cpd.decompose(tenL, incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_rank = (2, 3)
            cpd.decompose(tenL, incorrect_rank)

    def test_converged(self):
        """ Tests for converged method """
        tol = 0.01
        cpd = Parafac2(tol=tol)

        # when it is empty, which is the case at the object creation
        assert not cpd.converged

        # requires at least two values
        cpd.cost = [0.001]
        assert not cpd.converged

        # difference greater then `tol`
        cpd.cost = [0.1, 0.2]
        assert not cpd.converged

        # checks only the last two values
        cpd.cost = [0.0001, 0.0002, 0.1, 0.2]
        assert not cpd.converged

        cpd.cost = [0.1, 0.100001]
        assert cpd.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        cpd = Parafac2()
        cpd.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed

