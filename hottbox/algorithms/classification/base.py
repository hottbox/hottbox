from inspect import signature


# TODO: add interface and for checking labels at the training and testing stages (by analogy with sklearn)
# TODO: override __repr__ method by analogy with sklearn and how base Decomposition interface.
class Classifier(object):
    """ General interface for all classes that describe classification algorithms.

    Parameters
    -------
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling ``fit``, and will slow down that method.
    verbose : bool
        Enable verbose output.
    """

    def __init__(self, probability, verbose):
        self.probability = probability
        self.verbose = verbose

    @property
    def name(self):
        """ Name of the classifier.

        Returns
        -------
        str
        """
        return self.__class__.__name__

    def set_params(self, **params):
        """ Set the parameters of this estimator.

        Returns
        -------
        self
        """
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError("Invalid parameter '{0}' for estimator {1}. "
                                 "Check the list of available parameters "
                                 "with `estimator.get_params().keys()`.".format(key,
                                                                                self.name
                                                                                )
                                 )
            setattr(self, key, value)
            valid_params[key] = value
        return self

    def get_params(self):
        """ Get parameters for this estimator.

        Returns
        -------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        params = dict()
        for name in self._get_param_names():
            value = getattr(self, name, None)
            params[name] = value
        return params

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

    @classmethod
    def _get_param_names(cls):
        """ Get parameter names for the estimator.

        Returns
        -------
        param_names : list[str]
            Parameter names extracted from the constructors signature.
        """
        # Introspect the constructor arguments
        init_signature = signature(cls.__init__)

        # Extract the constructor positional and keyword parameters excluding 'self'
        params = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind == p.POSITIONAL_OR_KEYWORD]

        # Get and sort names of parameters extracted from the constructor
        param_names = sorted([p.name for p in params])

        return param_names