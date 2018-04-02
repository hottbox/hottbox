import warnings
import numpy as np
from .base import Decomposition, svd
from ...core.structures import Tensor, TensorTKD, residual_tensor
from ...core.operations import unfold


class BaseTucker(Decomposition):

    def __init__(self, process, mode_description, verbose):
        super(BaseTucker, self).__init__()
        self.process = process
        self.mode_description = mode_description
        self.verbose = verbose

    @property
    def converged(self):
        raise NotImplementedError('Not implemented in base (BaseTucker) class')

    def decompose(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (BaseTucker) class')

    def _init_fmat(self, tensor, rank):
        raise NotImplementedError('Not implemented in base (BaseTucker) class')

    def plot(self):
        raise NotImplementedError('Not implemented in base (BaseTucker) class')


class HOSVD(BaseTucker):
    """ Higher Order Singular Value Decomposition.

    Parameters
    ----------
    process : tuple
        Specifies the order of modes to be processed. The factor matrices for the missing modes will be set to identity.
        If empty, then all modes are processed in the consecutive ascending order.
    mode_description : str
    verbose : bool
        If True, enable verbose output
    """

    def __init__(self,  process=(), mode_description='mode_hosvd', verbose=False) -> None:
        super(HOSVD, self).__init__(process=process,
                                    mode_description=mode_description,
                                    verbose=verbose)

    def decompose(self, tensor, rank):
        """ Performs tucker decomposition via Higher Order Singular Value Decomposition (HOSVD)

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired multilinear rank for the given `tensor`

        Returns
        -------
        tensor_tkd : TensorTKD
            Tucker representation of the `tensor`
        """
        fmat = [np.array([])] * tensor.order
        core = tensor.copy()
        # TODO: can add check for self.process here
        if not self.process:
            self.process = tuple(range(tensor.order))
        for mode in range(tensor.order):
            if mode not in self.process:
                fmat[mode] = np.eye(rank[mode])
                continue
            tensor_unfolded = unfold(tensor.data, mode)
            U, _, _, = svd(tensor_unfolded, rank[mode])
            fmat[mode] = U
            core.mode_n_product(U.T, mode=mode)
        tensor_tkd = TensorTKD(fmat=fmat, core=core)
        if self.verbose:
            residual = residual_tensor(tensor, tensor_tkd)
            print('Relative error of approximation = {}'.format(abs(residual.frob_norm / tensor.frob_norm)))
        return tensor_tkd

    @property
    def converged(self):
        warnings.warn(
            "The {} algorithm is not iterative algorithm.\n"
            "Returning default value (True).".format(self.name), RuntimeWarning
        )
        return True

    def _init_fmat(self, tensor, rank):
        print("The {} algorithm does not required initialisation of factor matrices")

    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))


class HOOI(BaseTucker):
    """ Higher Order Orthogonal Iteration Decomposition.

    Parameters
    ----------
    init : str
        Type of factor matrix initialisation. Available options are `hosvd`.
    process : tuple
        Specifies the order of modes to be processed. The factor matrices for the missing modes will be set to identity.
        If empty, then all modes are processed in the consecutive ascending order.
        Note, initialisation of a factor matrix that corresponds to the mode at the first position is skipped.
    max_iter : int
        Maximum number of iteration
    epsilon : float
        Threshold for the relative error of approximation.
    tol : float
        Threshold for convergence of factor matrices
    random_state : int
    mode_description : str
    verbose : bool
        If True, enable verbose output

    Attributes
    ----------
    cost : list
        A list of relative approximation errors at each iteration of the algorithms.
    """

    def __init__(self, init='hosvd', max_iter=50, epsilon=10e-3, tol=10e-5,
                 random_state=None, process=(), mode_description='mode_hooi', verbose=False) -> None:
        super(HOOI, self).__init__(process=process,
                                   mode_description=mode_description,
                                   verbose=verbose)
        self.init = init
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tol = tol
        self.random_state = random_state
        # Initialise attributes
        self.cost = []

    def decompose(self, tensor, rank):
        """ Performs tucker decomposition via Higher Order Orthogonal Iteration (HOOI)

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired multilinear rank for the given `tensor`

        Returns
        -------
        tensor_tkd : TensorTKD
            Tucker representation of the `tensor`
        """
        tensor_tkd = None
        fmat_hooi = self._init_fmat(tensor, rank)
        norm = tensor.frob_norm
        if not self.process:
            self.process = tuple(range(tensor.order))
        for n_iter in range(self.max_iter):

            # Update factor matrices
            for i in self.process:
                tensor_approx = tensor.copy()
                for mode, fmat in enumerate(fmat_hooi):
                    if mode == i:
                        continue
                    tensor_approx.mode_n_product(fmat.T, mode=mode)
                fmat_hooi[i], _, _ = svd(tensor_approx.unfold(i).data, rank=rank[i])

            # Update core
            core = tensor.copy()
            for mode, fmat in enumerate(fmat_hooi):
                core.mode_n_product(fmat.T, mode=mode)

            # Update cost
            tensor_tkd = TensorTKD(fmat=fmat_hooi, core=core)
            residual = residual_tensor(tensor, tensor_tkd)
            self.cost.append(abs(residual.frob_norm / norm))
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
        return tensor_tkd

    @property
    def converged(self):
        """ Checks convergence of the HOOI algorithm.

        Returns
        -------
        bool
        """
        try:  # This insures that the cost has been computed at least twice without checking number of iterations
            is_converged = abs(self.cost[-2] - self.cost[-1]) <= self.tol
        except IndexError:
            is_converged = False
        return is_converged

    def _init_fmat(self, tensor, rank):
        """ Initialisation of factor matrices

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired multilinear rank for the given `tensor`

        Returns
        -------
        fmat : list[np.ndarray]
            List of factor matrices
        """
        self.cost = []  # Reset cost every time when method decompose is called
        if self.init is 'hosvd':
            hosvd = HOSVD(process=self.process[1:])
            tensor_hosvd = hosvd.decompose(tensor, rank)
        else:
            raise NotImplementedError('The given initialization ({}) is not available'.format(self.init))
        fmat = tensor_hosvd.fmat
        return fmat

    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))
