import warnings
import numpy as np
from .base import Decomposition, svd
from ...core.structures import Tensor, TensorTKD, residual_tensor
from ...core.operations import unfold


class BaseTucker(Decomposition):

    def __init__(self, process, verbose):
        super(BaseTucker, self).__init__()
        self.process = process
        self.verbose = verbose

    def copy(self):
        """ Copy of the Decomposition as a new object """
        new_object = super(BaseTucker, self).copy()
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(BaseTucker, self).name
        return decomposition_name

    @property
    def converged(self):
        raise NotImplementedError('Not implemented in base (BaseTucker) class')

    def decompose(self, tensor, rank, keep_meta):
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
    verbose : bool
        If True, enable verbose output
    """

    def __init__(self,  process=(), verbose=False) -> None:
        super(HOSVD, self).__init__(process=process,
                                    verbose=verbose)

    def copy(self):
        """ Copy of the HOSVD algorithm as a new object """
        new_object = super(HOSVD, self).copy()
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(HOSVD, self).name
        return decomposition_name

    def decompose(self, tensor, rank, keep_meta=0):
        """ Performs tucker decomposition via Higher Order Singular Value Decomposition (HOSVD)

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired multilinear rank for the given `tensor`
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor_tkd : TensorTKD
            Tucker representation of the `tensor`
        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be an object of `Tensor` class!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if tensor.order != len(rank):
            raise ValueError("Parameter `rank` should be tuple of the same length as the order of a tensor:\n"
                             "{} != {} (tensor.order != len(rank))".format(tensor.order, len(rank)))
        fmat = [np.array([])] * tensor.order
        core = tensor.copy()
        # TODO: can add check for self.process here
        if not self.process:
            self.process = tuple(range(tensor.order))
        for mode in range(tensor.order):
            if mode not in self.process:
                fmat[mode] = np.eye(tensor.shape[mode])
                continue
            tensor_unfolded = unfold(tensor.data, mode)
            U, _, _, = svd(tensor_unfolded, rank[mode])
            fmat[mode] = U
            core.mode_n_product(U.T, mode=mode)
        tensor_tkd = TensorTKD(fmat=fmat, core_values=core.data)
        if self.verbose:
            residual = residual_tensor(tensor, tensor_tkd)
            print('Relative error of approximation = {}'.format(abs(residual.frob_norm / tensor.frob_norm)))

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(tensor.modes)}
            tensor_tkd.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor_tkd.copy_modes(tensor)
        else:
            pass

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
    verbose : bool
        If True, enable verbose output

    Attributes
    ----------
    cost : list
        A list of relative approximation errors at each iteration of the algorithms.
    """

    def __init__(self, init='hosvd', max_iter=50, epsilon=10e-3, tol=10e-5,
                 random_state=None, process=(), verbose=False) -> None:
        super(HOOI, self).__init__(process=process,
                                   verbose=verbose)
        self.init = init
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tol = tol
        self.random_state = random_state
        # Initialise attributes
        self.cost = []

    def copy(self):
        """ Copy of the HOSVD algorithm as a new object """
        new_object = super(HOOI, self).copy()
        new_object.cost = []
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(HOOI, self).name
        return decomposition_name

    def decompose(self, tensor, rank, keep_meta=0):
        """ Performs tucker decomposition via Higher Order Orthogonal Iteration (HOOI)

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Desired multilinear rank for the given `tensor`
        keep_meta : int
            Keep meta information about modes of the given `tensor`.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices

        Returns
        -------
        tensor_tkd : TensorTKD
            Tucker representation of the `tensor`
        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be an object of `Tensor` class!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if tensor.order != len(rank):
            raise ValueError("Parameter `rank` should be tuple of the same length as the order of a tensor:\n"
                             "{} != {} (tensor.order != len(rank))".format(tensor.order, len(rank)))
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
            tensor_tkd = TensorTKD(fmat=fmat_hooi, core_values=core.data)
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

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(tensor.modes)}
            tensor_tkd.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor_tkd.copy_modes(tensor)
        else:
            pass

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
