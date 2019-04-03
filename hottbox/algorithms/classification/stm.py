import numpy as np
from ...core import Tensor
from .base import Classifier


class LSSTM(Classifier):
    """ Least Squares Support Tensor Machine (LS-STM) for binary classification.

    Parameters
    ----------
    C : float
        Penalty parameter of the error term.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Hard limit on iterations within solver.
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling ``fit``, and will slow down that method.
    verbose : bool
        Enable verbose output.

    Attributes
    ----------
    weights_ : list[np.ndarray]
        List of weights for each mode of the training data.
    bias_ : np.float64
    eta_history_ : np.ndarray
    bias_history_ : np.ndarray

    References
    ----------
    1)  Cichocki, Andrzej, et al. "Tensor networks for dimensionality reduction and large-scale optimization:
        Part 2 applications and future perspectives." Foundations and Trends in Machine Learning 9.6 (2017): 431-673.
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

    def fit(self, X, y):
        """ Fit the LS-STM model according to the given data.

        Parameters
        ----------
        X : list[Tensor]
            List of training samples of the same order and size.
        y : np.ndarray
            Target values relative to X for classification.

        Returns
        -------
        self
        """
        self._assert_train_data(X=X, y=y)

        # Convert binary labels to -1 and 1 (provides better results then 1 and 0)
        # This also allows to pass labels as stings
        self._orig_labels = list(set(y))
        y = np.array([1 if x == self._orig_labels[0] else -1 for x in y])

        # Initialise weights
        self.weights_ = [np.random.randn(dim) for dim in X[0].shape]

        eta_history = []
        bias_history = []
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
        y_pred : np.ndarray
            Class labels for samples in X.
        """
        self._assert_test_data(X=X)
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
        """ Compute probabilities of possible outcomes for samples in the provided data. """
        self._assert_test_data(X=X)
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
        self._assert_test_data(X=X, y=y)
        if not np.unique(y) in np.unique(self._orig_labels):
            raise ValueError("Provided set of labels is inconsistent with the those this {} instance was trained on!!!"
                             "Got {}, whereas {} expected".format(self.name,
                                                                 np.unique(y),
                                                                 self._orig_labels)
                             )

        y_pred = self.predict(X)
        acc = np.sum(y_pred == y) / y.size
        return acc

    def set_params(self, **params):
        super(LSSTM, self).set_params(**params)

    def get_params(self):
        return super(LSSTM, self).get_params()

    # TODO: Implementation of data input validation should be generalised.
    #  Data/model is validated twice when `score` method is called since it
    #  is based on `predict` method.
    def _assert_train_data(self, X, y):
        """ Validate train data

        Parameters
        ----------
        X : list[Tensor]
            List of multi-dimensional training samples.
        y : np.ndarray
            List of corresponding labels.
        """
        self._assert_data_samples(X)
        self._assert_samples_vs_labels(X, y)
        self._assert_data_labels(y)

    def _assert_test_data(self, X, y=None):
        """ Validate test data

        Parameters
        ----------
        X : list[Tensor]
            List of multi-dimensional test samples.
        y : np.ndarray
            True labels for test samples.
        """
        if self.weights_ is None:
            raise ValueError("This {} instance is not fitted yet. Call 'fit' with "
                             "appropriate arguments before using this method.".format(self.name))

        self._assert_data_samples(X)
        if y is not None:
            self._assert_samples_vs_labels(X=X, y=y)
            self._assert_data_labels(y)

        # Check that all samples in 'X' are of the same shape as during training.
        # By this point we have already made sure that samples in 'X' are of the same shape.
        weights_shape = tuple([w.shape[0] for w in self.weights_])
        if weights_shape != X[0].shape:
            raise ValueError("This {} instance has been trained of data of different shape. "
                             "Got {}, whereas {} is expected.".format(self.name,
                                                                      weights_shape,
                                                                      X[0].shape
                                                                      )
                             )

    @staticmethod
    def _assert_data_samples(X):
        """ Checks if all samples have same shape and order.

        Parameters
        ----------
        X : list[Tensor]
            List of multi-dimensional data samples.
        """
        if not isinstance(X, list):
            raise TypeError("All data samples should be passed as a list")

        if not all([isinstance(sample, Tensor) for sample in X]):
            raise TypeError("All data samples should be of `Tensor` class")

        if not all([sample.order == X[0].order for sample in X]):
            raise ValueError("All data samples should be of the same order")

        if not all([sample.shape == X[0].shape for sample in X]):
            raise ValueError("All data samples should be of the same shape")

    @staticmethod
    def _assert_data_labels(y):
        """ Checks if labels form a binary set.

        Parameters
        ----------
        y : np.ndarray
            List of labels for training data.
        """
        if np.unique(y).size > 2:
            raise ValueError("This is a binary classifier. Provided labels do not form a binary set")

    @staticmethod
    def _assert_samples_vs_labels(X, y):
        if y.size != len(X):
            raise ValueError("Number of provided labels should be equal to a number of data samples."
                             "Got {} labels, whereas {} is expected".format(y.size,
                                                                            len(X)
                                                                            )
                             )

    def _compute_eta(self, skip_mode):
        """ Compute an upper bound of margin errors.

        Parameters
        ----------
        skip_mode : int
            The mode for which LS-STM optimisation problem needs to be solved.

        Returns
        -------
        eta : np.float64
            An upper bound of margin errors.
        """
        eta = 1
        for mode in range(len(self.weights_)):
            if mode != skip_mode:
                eta *= (np.linalg.norm(self.weights_[mode]) ** 2)
        return eta

    def _compute_X_m(self, X, skip_mode):
        """ Contract multi-dimensional data samples with weights.

        Parameters
        ----------
        X : list[Tensor]
            List of all multi-dimensional data samples.
        skip_mode : int
            The mode for which contraction is skipped.

        Returns
        -------
        X_m : np.ndarray
            Matrix of contracted samples with weights along all modes except the ``skip_mode``.
            Has a shape ``(M, N)`` where ``M = len(X)`` and ``N = X[0].shape[skip_mode]``
        """
        X_m = np.zeros((len(X), X[0].shape[skip_mode]))
        for i, tensor in enumerate(X):
            temp = tensor.copy()
            for mode in range(X[0].order):
                if mode != skip_mode:
                    # TODO: this could be simplified when 'hottbox' will support mode-n product with a vector
                    temp.mode_n_product(np.expand_dims(self.weights_[mode], axis=0), mode=mode, inplace=True)
            X_m[i, :] = temp.data.squeeze()
        return X_m

    # TODO: potentially can be put in a separate module in order to be reused
    def _ls_optimizer(self, X_m, eta, labels):
        """ Solves LS-STM optimization problem for mode-n.

        Parameters
        ----------
        X_m : np.ndarray
            Matrix of contracted tensor-samples with weights along all
            modes except the current one.
        eta : np.float64
            An upper bound of margin errors.
        labels : np.ndarray
            The labels of the training data.

        Returns
        -------
        weights : np.ndarray
            Weights assigned to the features with respect to mode-n.
        bias : np.float64
            Constant in decision function.

        Notes
        -----
        Formulation of LS-STM for mode-n is equivalent to formulation of Least Squares
        Support Vector Machine.
        """
        M = X_m.shape[0]
        ones_row = np.ones(M)
        ones_col = np.expand_dims(ones_row, axis=1)
        identity = np.identity(M)

        # Kernel matrix (Linear for now)
        omega = np.dot(X_m, X_m.transpose())

        # We flip this around to simplify expression for matrix construction
        gamma = eta / self.C

        # Constructing a linear system (LHS x [bias, weights]^T = RHS)
        top = np.hstack([0, ones_row])
        bottom = np.hstack([ones_col, omega + gamma * identity])
        left_hand_side = np.vstack([top, bottom])

        right_hand_side = np.expand_dims(np.hstack([0, labels]), axis=1)

        # Obtain Lagrange multipliers
        alphas = np.dot(np.linalg.inv(left_hand_side), right_hand_side)

        weights = np.sum(alphas[1:, :] * X_m, axis=0)
        bias = alphas[0, 0]

        return weights, bias
