import numpy as np
from ...core.structures import BaseTensorTD


# TODO: add interface and for checking labels at the training and testing stages (by analogy with sklearn)
class Classifier(object):
    """ General interface for all classes that describe classification algorithms.

    Parameters
    -------
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    verbose : bool
        Enable verbose output.
    """

    def __init__(self, probability, verbose):
        self.probability = probability
        self.verbose = verbose

    @property
    def name(self):
        """ Name of the classifier

        Returns
        ----------
        str
        """
        return self.__class__.__name__

    def fit(self, X, y):
        """ Fit specified classification model according to the given training data.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation
        y : np.ndarray
            Target relative to X for classification
        """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    def predict(self, X):
        """ Perform classification on samples in X.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of test samples each of which is represented through a tensor factorisation
        """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    def predict_proba(self, X):
        """ Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation
        """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of test samples each of which is represented through a tensor factorisation
        y : np.ndarray
            True labels for X.
        """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    def grid_search(self, X, y, search_params, cv_params, inplace, n_jobs):
        """ Perform hyper parameter search with cross-validation for all base classifiers.

        Parameter setting that gave the best results on the hold out data are assigned to the base classifiers

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation
        y : np.ndarray
            Target relative to X for classification
        search_params : list[dict]
            List of dictionaries with parameters names (string) as keys and lists of parameter settings to try as values
        cv_params : dict
            Dictionary with kwargs that determine the cross-validation splitting strategy.
        inplace : bool
            If True, assign parameter setting that gave the best results on the hold out data to the base classifier
        n_jobs : int
            Number of jobs to run in parallel
        """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        if not self.probability:
            raise AttributeError('`predict_proba` is not available when `probability=False`')
