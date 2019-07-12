"""
Tests for the cmtf module
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
from hottbox.algorithms.decomposition.fusion.cmtf import CMTF

# TODO: make a base Testcmtf class to inherent from
class TestCMTF:
    """ Tests for CpRand class """
    def test_init(self):
        """ Tests for the constructor of cmtf algorithm """
        max_iter = 50
        epsilon = 10e-3
        tol = 10e-5
        verbose = False
        sample_size = None
        cmtf = CMTF(max_iter=max_iter,
                       epsilon=epsilon,
                       tol=tol,
                       verbose=verbose)
        assert not cmtf.cost         # check that this list is empty
        assert cmtf.name == CMTF.__name__
        assert cmtf.max_iter == max_iter
        assert cmtf.epsilon == epsilon
        assert cmtf.tol == tol
        assert cmtf.verbose == verbose

    def test_copy(self):
        """ Tests for copy method """
        cmtf = CMTF()
        cmtf.cost = [1, 2]
        cmtf_copy = cmtf.copy()

        assert cmtf_copy is not cmtf
        assert cmtf_copy.name == cmtf.name
        assert cmtf_copy.init == cmtf.init
        assert cmtf_copy.max_iter == cmtf.max_iter
        assert cmtf_copy.epsilon == cmtf.epsilon
        assert cmtf_copy.tol == cmtf.tol
        assert cmtf_copy.verbose == cmtf.verbose
        assert cmtf_copy.cost != cmtf.cost

        cmtf.max_iter += 1
        cmtf.epsilon += 1
        cmtf.tol += 1
        cmtf.verbose = not cmtf.verbose
        cmtf.cost = [3, 4]
        assert cmtf_copy.max_iter != cmtf.max_iter
        assert cmtf_copy.epsilon != cmtf.epsilon
        assert cmtf_copy.tol != cmtf.tol
        assert cmtf_copy.verbose != cmtf.verbose
        assert cmtf.cost != cmtf_copy.cost

    def test_init_fmat(self):
        """ Tests for _init_fmat method """
        np.random.seed(0)
        K = 3
        I_k = np.random.randint(4,15,K)
        J_k = np.random.randint(4,15,K)
        rank = (min(I_k)-1,)
        cmtf = CMTF()

        # ------ tests that cmtf.cost is reset each time _init_fmat is called
        cmtf.cost = [1, 2, 3]
        cmtf._init_fmat(I_k, J_k, rank)
        assert not cmtf.cost

        # ------ correct shape and type for factor matrices
        # svd type initialisation should produce factor matrices with orthogonal columns
        r = rank[0]
        A, B = cmtf._init_fmat(I_k, J_k, rank)
        for i, (a, b) in enumerate(zip(A,B)):
            assert a.shape == (I_k[i], r)
            assert b.shape == (J_k[i], r)

        t_k = np.random.randn(K) 
        #  ------ test for incorrect list
        with pytest.raises(TypeError):
            cmtf._init_fmat(t_k, J_k, rank)
       
       # ------ test for rank: expected to warn
        # Rank specified should be the match the specified shape
        rank = I_k[0]+1
        with pytest.warns(RuntimeWarning):
            cmtf._init_fmat(I_k, J_k, (rank,))

        #  ------ test for incorrect rank type 
        rank = I_k[0]
        with pytest.raises(IndexError):
            cmtf._init_fmat(I_k, J_k, rank)
    
    def test_decompose(self):
        """ Tests for decompose method """
        # ------ tests for termination conditions
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        np.random.seed(0)
        K = 3
        I_k = tuple(np.random.randint(4,15,K))
        rank = (min(I_k),)
        size = [(_a,rank[0]) for _a in I_k]
        y = [Tensor(np.random.randn(*sz)) for sz in size]
        tt = Tensor(np.random.randn(*I_k))
        cmtf = CMTF(verbose=True)

        # check for termination at max iter
        cmtf.max_iter = 10
        cmtf.epsilon = 0.01
        cmtf.tol = 0.0001
        cmtf.decompose(tt, y, rank)
        assert not cmtf.converged
        assert len(cmtf.cost) == cmtf.max_iter
        assert cmtf.cost[-1] > cmtf.epsilon

        # check for termination when acceptable level of approximation is achieved
        cmtf.max_iter = 10
        cmtf.epsilon = 1.6
        cmtf.tol = 10e-5
        cmtf.decompose(tt, y, rank)
        assert not cmtf.converged
        assert len(cmtf.cost) < cmtf.max_iter
        assert cmtf.cost[-1] <= cmtf.epsilon

        # check for termination at convergence
        cmtf.max_iter = 20
        cmtf.epsilon = 0.01
        cmtf.tol = 0.03
        cmtf.decompose(tt, y, rank)
        assert cmtf.converged
        assert len(cmtf.cost) < cmtf.max_iter
        assert cmtf.cost[-1] > cmtf.epsilon

        assert captured_output.getvalue() != ''  # to check that something was actually printed

        # ------ tests for correct output type and values
        cmtf = CMTF(max_iter=50, epsilon=10e-3, tol=10e-5)

        A, B, t_recon, m_recon = cmtf.decompose(tt, y, rank)
        # types
        assert isinstance(A, list)
        assert isinstance(B, list)
        assert isinstance(t_recon, Tensor)
        assert isinstance(m_recon, list)
        assert (all(isinstance(m, Tensor) for m in m_recon))
        
        # ------ tests that should FAIL due to wrong input type
        cmtf = CMTF()
        # tensor should be Tensor class
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            incorrect_tensor = np.arange(size).reshape(shape)
            correct_rank = (2,)
            y = [Tensor(np.random.randn(i,correct_rank[0])) for i in shape]
            cmtf.decompose(incorrect_tensor, y, correct_rank)
        # rank should be a tuple
        with pytest.raises(TypeError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = [2]
            y = [Tensor(np.random.randn(i,incorrect_rank[0])) for i in shape]
            cmtf.decompose(correct_tensor, y, incorrect_rank)
        # incorrect length of rank
        with pytest.raises(ValueError):
            shape = (5, 5, 5)
            size = reduce(lambda x, y: x * y, shape)
            correct_tensor = Tensor(np.arange(size).reshape(shape))
            incorrect_rank = (2, 3)
            y = [Tensor(np.random.randn(i,incorrect_rank[0])) for i in shape]
            cmtf.decompose(correct_tensor, y, incorrect_rank)

    def test_converged(self):
        """ Tests for converged method """
        tol = 0.01
        cmtf = CMTF(tol=tol)

        # when it is empty, which is the case at the object creation
        assert not cmtf.converged

        # requires at least two values
        cmtf.cost = [0.001]
        assert not cmtf.converged

        # difference greater then `tol`
        cmtf.cost = [0.1, 0.2]
        assert not cmtf.converged

        # checks only the last two values
        cmtf.cost = [0.0001, 0.0002, 0.1, 0.2]
        assert not cmtf.converged

        cmtf.cost = [0.1, 0.100001]
        assert cmtf.converged

    def test_plot(self):
        """ Tests for plot method """
        # This is only for coverage at the moment
        captured_output = io.StringIO()     # Create StringIO object for testing verbosity
        sys.stdout = captured_output        # and redirect stdout.
        cmtf = CMTF()
        cmtf.plot()
        assert captured_output.getvalue() != ''  # to check that something was actually printed
