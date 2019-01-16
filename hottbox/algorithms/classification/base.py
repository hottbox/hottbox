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

    def set_params(self, **params):
        """ Set the parameters of this estimator. """
        raise NotImplementedError('Not implemented in base ({}) class'.format(self.__class__.__name__))

    def get_params(self):
        """ Get parameters for this estimator. """
        raise NotImplementedError('Not implemented in base ({}) class'.format(self.__class__.__name__))

    def fit(self, X, y):
        """ Fit a classification model according to the given data. """
        raise NotImplementedError('Not implemented in base ({}) class'.format(self.__class__.__name__))

    def predict(self, X):
        """ Predict the class labels for the provided data. """
        raise NotImplementedError('Not implemented in base ({}) class'.format(self.__class__.__name__))

    def predict_proba(self, X):
        """ Compute probabilities of possible outcomes for samples in the provided data.. """
        raise NotImplementedError('Not implemented in base (Classifier) class')

    def score(self, X, y):
        """ Returns the mean accuracy on the given test data and labels. """
        raise NotImplementedError('Not implemented in base ({}) class'.format(self.__class__.__name__))
