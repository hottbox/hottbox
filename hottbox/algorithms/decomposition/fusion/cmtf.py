import functools
import warnings
import numpy as np
from hottbox.utils.generation.basic import residual_tensor
from hottbox.algorithms.decomposition.cpd import BaseCPD
from hottbox.core.structures import Tensor
from hottbox.core.operations import khatri_rao, hadamard, sampled_khatri_rao
from ..base import Decomposition, svd
from hottbox.utils.generation.basic import super_diag_tensor

# TODO: Organise this better - lazy work around used
class CMTF(BaseCPD):
    """ Coupled Matrix and Tensor factorization for two ``tensors`` of order n and 2 with respect to a specified ``rank``.
    Computed via alternating least squares (ALS)
    Parameters
    ----------
    max_iter : int
        Maximum number of iteration
    epsilon : float
        Threshold for the relative error of approximation.
    tol : float
        Threshold for convergence of factor matrices
    random_state : int
    verbose : bool
        If True, enable verbose output
    Attributes
    ----------
    cost : list
        A list of relative approximation errors at each iteration of the algorithm.
        
    References
    ----------
    
    ..  [1] Acar, Evrim, Evangelos E. Papalexakis, Gozde Gurdeniz, Morten A. Rasmussen, Anders J. Lawaetz, Mathias Nilsson and Rasmus Bro. “Structure-revealing data fusion.” BMC Bioinformatics (2013).
    ..  [2] Jeon, Byungsoo & Jeon, Inah & Sael, Lee & Kang, U. (2016). SCouT: Scalable coupled matrix-tensor factorization—Algorithm and discoveries. Int. Conf. Data Eng.. 811-822. 10.1109/ICDE.2016.7498292.
    """
    # TODO: change init use requiring a change in TensorCPD
    def __init__(self, max_iter=50, epsilon=10e-3, tol=10e-5,
                 random_state=None, verbose=False) -> None:
        super(CMTF, self).__init__(init='random',
                                  max_iter=max_iter,
                                  epsilon=epsilon,
                                  tol=tol,
                                  random_state=random_state,
                                  verbose=verbose)
        self.cost = []

    def copy(self):
        """ Copy of the CPD algorithm as a new object """
        new_object = super(CMTF, self).copy()
        new_object.cost = []
        return new_object

    @property
    def name(self):
        """ Name of the decomposition
        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(CMTF, self).name
        return decomposition_name

    def decompose(self, tensor, mlst, rank):
        """ Performs factorisation using ALS on the two instances of ``tensor`` with respect to the specified ``rank``
        Parameters
        ----------
        tensor : Tensor
            Multi-dimensional data to be decomposed
        matrix : List of `Tensor`
            List of two-dimensional `Tensor` to be decomposed
        rank : tuple
            Desired Kruskal rank for the given ``tensor``. Should contain only one value.
            If it is greater then any of dimensions then random initialisation is used
        kr_reverse : bool
        Returns
        -------
        tensor_cpd : TensorCPD
            CP representation of the ``tensor``
        Notes
        -----
        khatri-rao product should be of matrices in reversed order. But this will duplicate original data (e.g. images)
        Probably this has something to do with data ordering in Python and how it relates to kr product
        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be `Tensor`!")
        if not isinstance(mlst, list):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if len(rank) != 1:
            raise ValueError("Parameter `rank` should be tuple with only one value!")
        if not all(isinstance(m, Tensor) for m in mlst):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")
        if not all(m.order == 2 for m in mlst):
            raise ValueError("All elements of `mlst` should be of order 2. It is a list of matrices!")
        
        modes = np.array([list(m.shape) for m in mlst])
        N = len(modes)
        I, J = modes[:,0], modes[:,1]
        A, B = self._init_fmat(I, J, rank)
        norm = tensor.frob_norm
        for n_iter in range(self.max_iter):
            # Update tensor factors
            for i in range(N):
                V = hadamard([np.dot(a_i.T, a_i) for k, a_i in enumerate(A) if k != i])
                V += B[i].T.dot(B[i])
                kr_result = khatri_rao(A, skip_matrix=i, reverse=True)
                prod_a = np.concatenate([tensor.unfold(i, inplace=False).data, mlst[i].data], axis=1)
                prod_b = np.concatenate([kr_result.T, B[i].T], axis=1).T
                A[i] = prod_a.dot(prod_b).dot(np.linalg.pinv(V))
            for i in range(N):
                B[i] = mlst[i].data.T.dot(np.linalg.pinv(A[i]).T)
            
            t_recon, m_recon = self.reconstruct(A,B,N)
            
            residual = np.linalg.norm(tensor.data-t_recon.data)
            for i in range(N):
                residual += np.linalg.norm(mlst[i].data-m_recon[i].data)
            self.cost.append(abs(residual)/norm)
        
            if self.verbose:
                print('Iter {}: relative error of approximation = {}'.format(n_iter, self.cost[-1]))

            # Check termination conditions
            if self.cost[-1] <= self.epsilon:
                if self.verbose:
                    print('Relative error of approximation has reached the acceptable level: {}'.format(self.cost[-1]))
                break
            if self.converged:
                if self.verbose:
                    print('Converged in {} iteration(s)'.format(len(self.cost)))
                break
        
        if self.verbose and not self.converged and self.cost[-1] > self.epsilon:
            print('Maximum number of iterations ({}) has been reached. '
                  'Variation = {}'.format(self.max_iter, abs(self.cost[-2] - self.cost[-1])))

        # TODO: possibly make another structure
        return A, B, t_recon, m_recon

    @property
    def converged(self):
        """ Checks convergence of the CPD-ALS algorithm.
        Returns
        -------
        bool
        """
        try:  # This insures that the cost has been computed at least twice without checking number of iterations
            is_converged = abs(self.cost[-2] - self.cost[-1]) <= self.tol
        except IndexError:
            is_converged = False
        return is_converged

    def _init_fmat(self, I, J, rank):
        """ Initialisation of matrices used in CMTF
        Parameters
        ----------
        I : np.ndarray(int)
            Shape[0] of all matrices
        J : np.ndarray(int)
            Shape[1] of all matrices
        rank : int
            The rank specified for factorisation
        Returns
        -------
        (A,B) : List[np.ndarray]
            Two lists of the factor matrices
        """
        self.cost = []  # Reset cost every time when method decompose is called 
        R = rank[0]
        if (np.array(I) < R).sum() != 0:
            warnings.warn(
                "Specified rank value is greater then one of the dimensions of a tensor ({} > {}).\n"
                "Factor matrices have been initialized randomly.".format(R, I), RuntimeWarning
            )
        A = [np.random.randn(i_n, R) for i_n in I]
        B = [np.random.randn(j_n, R) for j_n in J]
        return A, B

    def reconstruct(self, A, B, N, keep_meta=0):
        """ Reconstruct the tensor and matrix after the coupled factorisation
        Parameters
        ----------
        A : np.ndarray
            Multidimensional data obtained from the factorisation
        B : np.ndarray
            Multidimensional data obtained from the factorisation
        N : int
            Number of matrices provided to fuse
        Returns
        -------
        (H,V,S,U) : Tuple[np.ndarray]
            Matrices used in CMTF
        """
        core_values = np.repeat(np.array([1]), A[0].shape[1])
        r = (A[0].shape[1],)
        core_shape = r * len(A)
        core_tensor = super_diag_tensor(core_shape, values=core_values)

        for mode, fmat in enumerate(A):
            core_tensor.mode_n_product(fmat, mode=mode, inplace=True)
        
        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(self.modes)}
            tensor.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor.copy_modes(self)
        else:
            pass

        lrecon = [Tensor(A[i].dot(B[i].T)) for i in range(N)]
    
        return core_tensor, lrecon
    
    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))
