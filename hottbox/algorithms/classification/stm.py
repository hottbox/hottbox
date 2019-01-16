import numpy as np
from ...core import Tensor
from .base import Classifier


class LSSTM(Classifier):
    """ Least Squares Support Tensor Machine (LS-STM) for binary classification.

    Parameters
    ----------
    C : float
        Penalty parameter C of the error term.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Hard limit on iterations within solver.
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    verbose : bool
        Enable verbose output.

    Attributes
    ----------
    weights_ : list[np.ndarray]
        List of weights for each mode of the training data
    bias_ : np.float64
    eta_history_ : np.ndarray
    bias_history_ : np.ndarray

    Notes
    -----
    [1] Zhao, Xinbin, et al. "Least squares twin support tensor machine for classification."
        Journal of Information & Computational Science 11.12 (2014): 4175-4189.

    [2] Cichocki, Andrzej, et al. "Tensor networks for dimensionality reduction and large-scale optimization:
        Part 2 applications and future perspectives."
        Foundations and Trends in Machine Learning 9.6 (2017): 431-673.

    """
    def __init__(self, C=1, tol=1e-3, max_iter=100, probability=False, verbose=False):
        super(LSSTM, self).__init__(probability=probability,
                                    verbose=verbose)
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.bias_ = None
        self.weights_ = None
        self.eta_history_ = None
        self.bias_history_ = None
        self._orig_labels = None

    def set_params(self, **params):
        super(LSSTM, self).set_params(**params)

    def get_params(self):
        return super(LSSTM, self).get_params()

    def fit(self, X, y):
        """ Fit the LS-STM model according to the given data.

        Parameters
        ----------
        X : list[Tensor]
            List of training samples of the same order and size.
        y : np.ndarray
            Target values relative to X for classification.
            of length M of labels +1, -1

        Returns
        -------
        self : object
        """
        self._assert_data_samples(X)
        self._assert_data_labels(y)

        # Binaries labels
        self._orig_labels = list(set(y))
        y = np.array([1 if x == self._orig_labels[0] else -1 for x in y])

        # Initialise weights
        self.weights_ = [np.random.randn(dim) for dim in X[0].shape]

        eta_history = []
        bias_history = [0]
        for n_iter in range(self.max_iter):
            eta_iter = []
            bias = 0  # no need to track bias for different modes
            for mode_n in range(X[0].order):
                X_m = self._compute_X_m(X, skip_mode=mode_n)
                eta = self._compute_eta(skip_mode=mode_n)
                eta_iter.append(eta)

                # Update LS-STM model weights on the fly
                self.weights_[mode_n], bias = self._ls_optimizer(X_m, eta, y)

            # Extend history for and check for convergence
            eta_history.append(eta_iter)
            bias_history.append(bias)
            if n_iter > 10:
                err1 = np.diff(eta_history[-2:], axis=0)
                err2 = bias_history[-2] - bias_history[-1]
                if np.all(np.abs(err1) <= self.tol) and abs(err2) <= self.tol:
                    break

        self.bias_ = bias_history[-1]
        self.bias_history_ = np.array(bias_history)
        self.eta_history_ = np.array(eta_history)
        return self

    def predict(self, X):
        """ Predict the class labels for the provided data.

        Parameters
        ----------
        X : list[Tensor]
            List of test samples.

        Returns
        -------
        y_pred : list[np.ndarray]
            Class labels for samples in X.
        """
        self._assert_data_samples(X)
        if self.weights_ is None:
            raise ValueError("This {} instance is not fitted yet. Call 'fit' with "
                             "appropriate arguments before using this method.".format(self.name))

        # Check that all samples in 'X' are of the same shape as during training.
        # By this point we have made sure that samples in 'X' are of the same shape.
        weights_shape = tuple([w.shape[0] for w in self.weights_])
        if not all([w_shape == X[0].shape[mode] for mode, w_shape in enumerate(weights_shape)]):
            raise ValueError("This {} instance has been trained of data of different shape. "
                             "Got {}, whereas {} is expected.".format(self.name,
                                                                      weights_shape,
                                                                      X[0].shape
                                                                      )
                             )
        y_pred = []
        for test_sample in X:
            temp = test_sample.copy()
            for mode_n in range(X[0].order):
                # TODO: this could be simplified when 'hottbox' will support mode-n product with a vector
                temp.mode_n_product(np.expand_dims(self.weights_[mode_n], axis=0), mode=mode_n, inplace=True)
            y_pred.append(np.sign(temp.data.squeeze() + self.bias_))

        y_pred = np.array([self._orig_labels[0] if x == 1 else self._orig_labels[1] for x in y_pred])
        return y_pred

    def predict_proba(self, X):
        return super(LSSTM, self).predict_proba(X=X)

    def score(self, X, y):
        """ Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : list[Tensor]
            List of test samples.
        y : np.ndarray
            True labels for test samples.

        Returns
        -------
        acc : np.float64
            Mean accuracy of ``self.predict(X)`` with respect to ``y``.
        """
        y_pred = self.predict(X)
        acc = np.sum(y_pred == y) / y.size
        return acc

    @staticmethod
    def _assert_data_samples(X):
        """ Checks if all samples have same shape and order.

        Parameters
        ----------
        X : list[Tensor]
            List of data samples of ``Tensor`` class.
        """
        if not isinstance(X, list):
            raise TypeError("All data samples should be passed as a list")

        if not all([isinstance(_, Tensor) for _ in X]):
            raise TypeError("All data samples should be of `Tensor` class")

        if not all([_.order for _ in X]):
            raise ValueError("All data samples should be of the same order")

        if not all([_.shape for _ in X]):
            raise ValueError("All data samples should be of the same shape")

    @staticmethod
    def _assert_data_labels(y):
        """ Checks if labels form a binary set.

        Parameters
        ----------
        y : np.ndarray
            List of labels for training data.
        """
        if np.unique(y).size != 2:
            raise ValueError("LS-STM is a binary classifier. Provided labels do not form a binary set")

    def _compute_eta(self, skip_mode):
        """

        Parameters
        ----------
        skip_mode : int
            The mode for which LS-STM optimisation problem needs to be solved.

        Returns
        -------
        eta : np.float64
            Parameter to be used in LS-STM optimization problem
        """
        eta = 1
        for mode in range(len(self.weights_)):
            if mode != skip_mode:
                eta *= (np.linalg.norm(self.weights_[mode]) ** 2)
        return eta

    def _compute_X_m(self, X, skip_mode):
        """

        Parameters
        ----------
        X : list[Tensor]
            All the data as list of tensor objects.
        skip_mode : int
            The mode for which LS-STM optimisation problem needs to be solved.

        Returns
        -------
        X_m : np.ndarray
            Array to be used in LS-STM optimization problem.
            Has a shape ``(M, N)`` where ``M = len(X)`` and ``N = X[0].shape[skip_mode]``
        """
        X_m = np.zeros((len(X), X[0].shape[skip_mode]))
        for i, tensor in enumerate(X):
            temp = tensor.copy()
            for mode in range(X[0].order):
                if mode != skip_mode:
                    temp.mode_n_product(np.expand_dims(self.weights_[mode], axis=0), mode=mode, inplace=True)
            X_m[i, :] = temp.data.squeeze()
        return X_m

    # TODO: potentially can be put in a separate module in order to be reused
    def _ls_optimizer(self, X_m, eta, labels):
        """ Solves LS-STM optimization problem for mode-n

        Parameters
        ----------
        X_m : np.ndarray
            Matrix of contracted tensors along all weights except the current n
        eta : np.float64
            Parameter to be used in the algorithm
        labels : np.ndarray
            The labels of the training data

        Returns
        -------
        weights : np.ndarray
            Weights obtained by solving LS-STM optimization problem for mode-n
        bias : np.float64
            Bias obtained by solving LS-STM optimization problem for mode-n
        """
        M = X_m.shape[0]
        ones = np.ones(M)
        identity = np.identity(M)
        gamma = eta / self.C
        omega = np.dot(X_m, X_m.transpose())

        right_hand_side = np.expand_dims(np.hstack([0, labels]), axis=1)

        left_column = np.expand_dims(np.hstack([0, ones]), axis=1)
        right_block = np.vstack([ones, omega + gamma * identity])
        left_hand_side = np.hstack([left_column, right_block])

        alphas = np.dot(np.linalg.inv(left_hand_side), right_hand_side)

        weights = np.sum(alphas[1:, :] * X_m, axis=0)
        bias = alphas[0, 0]

        return weights, bias
