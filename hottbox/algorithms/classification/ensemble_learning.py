import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from ...core.structures import BaseTensorTD, TensorCPD, TensorTKD

from .base import Classifier


class BaseTensorEnsembleClassifier(Classifier):
    """
    This is general interface for all classes that describe tensor ensemble learning algorithms.

    Parameters
    -------
    base_clf : list[SklearnClassifier]
        List of classifiers
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    verbose : bool
        Enable verbose output.
    """

    def __init__(self, base_clf, probability, verbose):
        self._validate_input_params(base_clf=base_clf, probability=probability)
        self._sync_base_clf_probability(base_clf=base_clf, probability=probability)
        super(BaseTensorEnsembleClassifier, self).__init__(probability=probability,
                                                           verbose=verbose)
        self.base_clf = base_clf

    def __str__(self):
        self_as_string = "{}(base_clf={}, probability={}".format(self.name,
                                                                 self.name_base_clf,
                                                                 self.probability)
        return self_as_string

    def __repr__(self):
        return str(self)

    @staticmethod
    def _sync_base_clf_probability(base_clf, probability):
        """ Enable or disable probability estimation for the base classifiers

        Parameters
        ----------
        base_clf : list[SklearnClassifier]
        probability : bool
        """
        for clf in base_clf:
            # clf.set_params(probability=probability)
            # Apparently, not all sklearn classes require to set probability  to `True`
            # TODO: throw a warning?
            if hasattr(clf, 'probability'):
                clf.set_params(probability=probability)

    @staticmethod
    def _validate_input_params(base_clf, probability):
        """ Validate input parameters for the TEL constructor

        Parameters
        ----------
        base_clf : list[SklearnClassifier]
        """
        if not isinstance(base_clf, list):
            raise TypeError('Input parameter `base_clf` should be a list')

        # Check that all base classifiers have required api
        if probability:
            required_api = {'fit', 'predict', 'predict_proba'}
        else:
            required_api = {'fit', 'predict'}
        for i, clf in enumerate(base_clf):
            all_attr = dir(clf)
            if not required_api.issubset(all_attr):
                missing_api = [api for api in required_api if api not in all_attr]
                raise ValueError('Base classifier #{}: does not support required API ({})!!!\n'
                                 'Missing methods are: {}'.format(i, required_api, missing_api))

    @property
    def name_base_clf(self):
        """ Get names of all employed base classifiers

        Returns
        -------
        names : list[str]
        """
        names = [clf.__class__.__name__ for clf in self.base_clf]
        return names

    def get_params_base_clf(self, i):
        """ Get parameters of an employed base classifier

        Parameters
        ----------
        i : int
            Specifies a base classifier which parameters are or interest

        Returns
        -------
        base_clf_params : dict
        """
        base_clf_params = self.base_clf[i].get_params()
        return base_clf_params

    def set_params_base_clf(self, params, i):
        """ Set parameters of an employed base classifier

        Parameters
        ----------
        params : dict
            Dictionary with parameters for the base classifier
        i : int
            Specifies a base classifier for which the parameters are to be changed
        """
        self.base_clf[i].set_params(**params)

    def fit(self, X, y):
        """ Train all base classifiers at once

        Parameters
        ----------
        X : list[BaseTensorTD]
        y : np.ndarray

        Returns
        -------
        self
        """
        X_list = self.decomp_to_array(X)
        for i, X_new in enumerate(X_list):
            self.fit_base_clf(X=X_new,
                              y=y,
                              clf_num=i)
        return self

    def fit_base_clf(self, X, y, clf_num):
        """ Train specific base classifier

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : np.ndarray
            Target relative to X for classification
        clf_num : int
            Positional number of the base classifier to be used

        Returns
        ----------
        """
        if self.verbose:
            print('Base classifier #{} ({}): Learning model parameters'.format(clf_num, self.name_base_clf[clf_num]))
        self.base_clf[clf_num].fit(X, y)
        return self

    def predict(self, X):
        X_list = self.decomp_to_array(X)
        all_labels = np.empty((len(X), len(X_list)), dtype=np.int8)
        for i, X_new in enumerate(X_list):
            all_labels[:, i] = self.predict_base_clf(X=X_new,
                                                     clf_num=i)
        # Find the most frequent values along each row of a matrix of labels from each base classifier
        axis = 1
        u, indices = np.unique(all_labels, return_inverse=True)
        y_pred = u[np.argmax(np.apply_along_axis(np.bincount,
                                                 axis,
                                                 indices.reshape(all_labels.shape),
                                                 None,
                                                 np.max(indices) + 1),
                             axis=axis
                             )]
        return y_pred

    def predict_base_clf(self, X, clf_num):
        """ Perform classification on samples in X on a specific base classifier

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Test data, where n_samples is the number of samples and n_features is the number of features.
        clf_num : int
            Positional number of the base classifier is to be used

        Returns
        -------
        np.ndarray
            Class labels for samples in X.
        """
        return self.base_clf[clf_num].predict(X)

    def predict_proba(self, X):
        X_list = self.decomp_to_array(X)
        n_estimators = len(X_list)
        rows = X_list[0].shape[0]
        cols = self.base_clf[0].classes_.size
        y_pred_proba = np.zeros((rows, cols))
        for i, X_new in enumerate(X_list):
            y_pred_proba += self.predict_proba_base_clf(X=X_new,
                                                        clf_num=i)
        y_pred_proba = np.divide(y_pred_proba, n_estimators)
        return y_pred_proba

    def predict_proba_base_clf(self, X, clf_num):
        """ Compute probabilities of possible outcomes for samples in X on a specific base classifier

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Test data, where n_samples is the number of samples and n_features is the number of features.
        clf_num : int
            Positional number of the base classifier is to be used

        Returns
        -------
        np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `base_clf[clf_num].classes_`.
        """
        self._check_proba()
        return self.base_clf[clf_num].predict_proba(X)

    def score(self, X, y):
        """ Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : list[BaseTensorTD]
        y : np.ndarray

        Returns
        -------
        acc = np.ndarray
        """
        if self.probability:
            pred_proba = self.predict_proba(X)
            y_pred = self._proba_to_label(pred_proba)
        else:
            y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        return acc

    def grid_search(self, X, y, search_params, cv_params=None, inplace=True, n_jobs=-1):
        if not isinstance(search_params, list):
            raise TypeError('Input parameter `search_params` should be a list of grid params')
        if len(search_params) != len(self.base_clf):
            raise ValueError('Wrong number of searching params for the hyperparameter tuning of all base classifiers!!!\n'
                             '{} dicts of searching params are required, '
                             'whereas, {} dicts have been provided'.format(len(self.base_clf), len(search_params)))
        X_new = self.decomp_to_array(X)
        best_params_list = []
        for i in range(0, len(X_new)):
            best_params = self.grid_search_base_clf(X=X_new[i],
                                                    y=y,
                                                    search_params=search_params[i],
                                                    clf_num=i,
                                                    cv_params=cv_params,
                                                    inplace=inplace,
                                                    n_jobs=n_jobs)
            best_params_list.append(best_params)
        return best_params_list

    # TODO: not sure whether default values is the good idea here
    def grid_search_base_clf(self, X, y, search_params, clf_num, cv_params=None, inplace=True, n_jobs=-1):
        """ Perform hyper parameter search with cross-validation for the specified base classifier.
        Parameter setting that gave the best results on the hold out data are assigned to this base classifier

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : np.ndarray
            Target relative to X for classification
        search_params : dict
            Parameters names (string) as keys and lists of parameter settings to try as values
        clf_num : int
            Positional number of the base classifier which hyperparameters are to be tuned
        cv_params : dict
            Dictionary with kwargs that determine the cross-validation splitting strategy.
        inplace : bool
            If True, assign parameter setting that gave the best results on the hold out data to the base classifier
        n_jobs : int
            Number of jobs to run in parallel

        Returns
        ----------
        dict
            Parameter setting that gave the best results on the hold out data
        """
        if self.verbose:
            print('Base classifier #{} ({}): Tuning the hyperparameters'.format(clf_num, self.name_base_clf[clf_num]))
        if not isinstance(search_params, dict):
            raise TypeError('Input parameter `search_params` should be a dict of parameters names (string) as keys '
                            'and lists of parameter settings to try as values')
        if cv_params is None:
            # Default parameters for cross-validation
            cv_params = dict(n_splits=1, test_size=0.2, random_state=42)
        cv = StratifiedShuffleSplit(**cv_params)
        grid = GridSearchCV(self.base_clf[clf_num],
                            param_grid=search_params,
                            cv=cv,
                            n_jobs=n_jobs
                            )
        grid.fit(X, y)
        if inplace:
            self.set_params_base_clf(grid.best_params_, clf_num)
        return grid.best_params_

    def _decomp_to_array(self, decomp_list):
        """ Construct new datasets based on factor vectors of each tensor factorisation from the `decomp_list`.

        Parameters
        ----------
        decomp_list : list[BaseTensorTD]
            List of tensor factorisations of all samples

        Notes
        -------
        This method should implement all manipulations with rearranging factor vectors. This should take into account
        the type of decomposition used prior the classification
        """
        raise NotImplementedError('Not implemented in base (BaseEnsembleClassifier) class')

    def decomp_to_array(self, decomp_list):

        data_list = self._decomp_to_array(decomp_list)
        if len(self.base_clf) != len(data_list):
            raise ValueError('Not enough base classifiers!!!\n'
                             'During object creation there had been specified {} base classifiers.\n'
                             'Whereas {} base classifiers are required for the {} classifier.'
                             ''.format(len(self.base_clf), len(data_list), self.name)
                             )
        return data_list

    def _proba_to_label(self, pred_proba):
        """ Assign label with respect to the highest probability among all classes in the model

        Parameters
        ----------
        pred_proba : np.ndarray
            Output of the self.predict_proba(). That is: probability of the sample for each class in the model.
            The columns correspond to the classes in sorted order, as they appear in the attribute
            `base_clf[i].classes_`.

        Returns
        -------
        y_pred : np.ndarray
        """
        df = pd.DataFrame(data=pred_proba, columns=self.base_clf[0].classes_)
        y_pred = df.idxmax(axis=1).as_matrix()
        return y_pred

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        if not self.probability:
            raise AttributeError('`predict_proba` is not available when `probability=False`')


class TelVI(BaseTensorEnsembleClassifier):
    """ Tensor Ensemble Learning: Vectors Independently (TelVI)

    Parameters
    ----------
    base_clf : list[SklearnClassifier]
        List of classifiers that will be used for the corresponding collection of the factor vectors of the tensor
        decomposition. This list does not have to be heterogeneous. However, all classifiers should support sklearn API.
        Length of this list should be equal to the number of collection of the factor vectors, otherwise an exception
        will be thrown. This is checked after the data have been splitted inside `decomp_to_array` method (called from
        `fit`, `predict`, `predict_proba`, `grid_search_base_clf`)
    probability : bool
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    verbose : bool
        Enable verbose output.

    References
    ----------
    -   Ilia Kisil, Ahmad Moniri, Danilo P. Mandic. "Tensor Ensemble Learning for Multidimensional Data."
        In 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP), pp. 1358-1362. IEEE, 2018.
    """
    def __init__(self, base_clf, probability=False, verbose=False):
        super(TelVI, self).__init__(base_clf=base_clf,
                                    probability=probability,
                                    verbose=verbose)

    def get_params_base_clf(self, i=-1):
        """ Get parameters of employed base classifier

        Parameters
        ----------
        i : int
            Positional number of the base classifier. By default outputs parameters for all base classifiers

        Returns
        -------
        Union[list[dict], dict]
        """
        if i < 0:
            return [super(TelVI, self).get_params_base_clf(i=clf_num) for clf_num in range(len(self.base_clf))]
        else:
            return super(TelVI, self).get_params_base_clf(i=i)

    def set_params_base_clf(self, params, i):
        """ Set parameters for the specified base classifier

        Parameters
        ----------
        i : int
            Positional number of the base classifier.
        params : dict
            Dictionary with parameters for the base classifier
        """
        super(TelVI, self).set_params_base_clf(params=params,
                                               i=i)

    def fit(self, X, y):
        """ Fit specified classification models according to the given training data.

        Parameters
        ----------
        X : {list[TensorCPD], list[TensorTKD]}
            List of training samples each of which is represented through a tensor factorisation
        y : np.ndarray
            Target relative to X for classification

        Returns
        -------
        self : object
        """
        super(TelVI, self).fit(X=X,
                               y=y)
        return self

    def predict(self, X):
        """ Perform classification on samples in X.

        Parameters
        ----------
        X : {list[TensorCPD], list[TensorTKD]}
            List of training samples each of which is represented through a tensor factorisation

        Returns
        -------
        y_pred : np.ndarray
            Class labels for samples in X.
        """
        y_pred = super(TelVI, self).predict(X=X)
        return y_pred

    def predict_proba(self, X):
        """ Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {list[TensorCPD], list[TensorTKD]}
            List of training samples each of which is represented through a tensor factorisation

        Returns
        -------
        y_pred_proba : np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `base_clf[i].classes_`.
        """
        y_pred_proba = super(TelVI, self).predict_proba(X=X)
        return y_pred_proba

    def score(self, X, y):
        acc = super(TelVI, self).score(X, y)
        return acc

    def grid_search(self, X, y, search_params, cv_params=None, inplace=True, n_jobs=-1):
        """ Perform hyper parameter search with cross-validation for all base classifiers. Parameter setting that gave
        the best results on the hold out data are assigned to the base classifiers

        Parameters
        ----------
        X : {list[TensorCPD], list[TensorTKD]}
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

        Returns
        -------
        best_params : list[dict]
            List of parameter setting that gave the best results on the hold out data for the corresponding classifier

        Raises
        -------
        TypeError
            If the searching parameters are not provided as a list
        ValueError
            If the searching parameters are not provided for each of the base classifier (lists length comparison)
            Note: All elements of `search_params` must contain only valid parameters for the respective base classifiers
        """
        best_params = super(TelVI, self).grid_search(X=X,
                                                     y=y,
                                                     search_params=search_params,
                                                     cv_params=cv_params,
                                                     inplace=inplace,
                                                     n_jobs=n_jobs)
        return best_params

    def decomp_to_array(self, decomp_list):
        data_list = super(TelVI, self).decomp_to_array(decomp_list=decomp_list)
        return data_list

    def _decomp_to_array(self, decomp_list):
        """ Combine respective column vectors of factor matrices for all tensor factorisations.

        Each factor vector is treated as sample. Corresponding factor matrices should be of the same shape

        Parameters
        ----------
        decomp_list : {list[TensorCPD], list[TensorTKD]}
            List of tensor factorisations of all samples

        Returns
        -------
        data_list : list[np.ndarray]

        Notes
        -------
        Each element of the `decomp_list` is the tensor factorisation of a sample. We do unfolding of tensor
        factorisation, i.e. unfold each factor matrix and stack them together in a long row vector. Then we stack these
        vectors into a matrix [sample \\times tensor features]. Then split this matrix in accordance with the lengths of
        factor vectors of the original decomposition
        """
        sample_decomp = decomp_list[0]
        row = np.array([])
        for fmat in sample_decomp.fmat:
            row = np.append(row, fmat.T.flatten())  # Transpose is crucial
        data = row

        for sample in range(1, len(decomp_list)):
            sample_decomp = decomp_list[sample]
            row = np.array([])
            for fmat in sample_decomp.fmat:
                row = np.append(row, fmat.T.flatten())  # Transpose is crucial
            data = np.vstack((data, row))

        # Create list of column indices for splitting different factor vectors
        split_idx = [0]  # this is the offset for the first split index
        for fmat in sample_decomp.fmat:
            i, j = fmat.shape
            split_idx = split_idx + [split_idx[-1] + i*x for x in range(1, j+1)]
        # Don't need the last index in order to avoid an empty element when np.hsplit is called
        split_idx = split_idx[1:-1]
        data_list = np.hsplit(data, split_idx)

        return data_list


class TelVAC(BaseTensorEnsembleClassifier):
    def __init__(self, base_clf, probability=True, verbose=True):
        super(TelVAC, self).__init__(base_clf=base_clf,
                                     probability=probability,
                                     verbose=verbose)

    def get_params_base_clf(self, i=-1):
        """ Get parameters of employed base classifier

        Parameters
        ----------
        i : int
            Positional number of the base classifier. By default outputs parameters for all base classifiers

        Returns
        -------
        Union[list[dict], dict]
        """
        if i < 0:
            return [super(TelVAC, self).get_params_base_clf(i=clf_num) for clf_num in range(len(self.base_clf))]
        else:
            return super(TelVAC, self).get_params_base_clf(i=i)

    def set_params_base_clf(self, params, i):
        """ Set parameters for the specified base classifier

        Parameters
        ----------
        i : int
            Positional number of the base classifier.
        params : dict
            Dictionary with parameters for the base classifier
        """
        super(TelVAC, self).set_params_base_clf(params=params,
                                                i=i)

    def fit(self, X, y):
        """ Fit specified classification models according to the given training data.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation
        y : np.ndarray
            Target relative to X for classification

        Returns
        -------
        self : object
        """
        super(TelVAC, self).fit(X=X,
                                y=y)
        return self

    def predict(self, X):
        """ Perform classification on samples in X.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation

        Returns
        -------
        y_pred : np.ndarray
            Class labels for samples in X.
        """
        y_pred = super(TelVAC, self).predict(X=X)
        return y_pred

    def predict_proba(self, X):
        """ Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : list[BaseTensorTD]
            List of training samples each of which is represented through a tensor factorisation

        Returns
        -------
        y_pred_proba : np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `base_clf[i].classes_`.
        """
        y_pred_proba = super(TelVAC, self).predict_proba(X=X)
        return y_pred_proba

    def score(self, X, y):
        acc = super(TelVAC, self).score(X, y)
        return acc

    def grid_search(self, X, y, search_params, cv_params=None, inplace=True, n_jobs=-1):
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

        Returns
        -------
        best_params : list[dict]
            List of parameter setting that gave the best results on the hold out data for the corresponding classifier

        Raises
        -------
        TypeError
            If the searching parameters are not provided as a list
        ValueError
            If the searching parameters are not provided for each of the base classifier (lists length comparison)
            Note: All elements of `search_params` must contain only valid parameters for the respective base classifiers
        """
        best_params = super(TelVAC, self).grid_search(X=X,
                                                      y=y,
                                                      search_params=search_params,
                                                      cv_params=cv_params,
                                                      inplace=inplace,
                                                      n_jobs=n_jobs)
        return best_params

    def decomp_to_array(self, decomp_list):
        data_list = super(TelVAC, self).decomp_to_array(decomp_list=decomp_list)
        return data_list

    def _decomp_to_array(self, decomp_list):
        """ Combine column vectors of factor matrices for all tensor factorisations.

        Each combination belongs to different dataset.
        Each factor vector is treated as sample. Corresponding factor matrices should be of the same shape

        Parameters
        ----------
        decomp_list : {list[TensorCPD], list[TensorTKD]}
            List of tensor factorisations of all samples

        Returns
        -------
        data_list : list[np.ndarray]
            List of new datasets of shape [n_samples, n_features]. Each sample is the combination of one factor vector
            from each of the factor matrices of the tensor factorisation for a given sample.
            len(data_list) = n_new_samples
            data_list[i].shape = [n_samples, n_new_features]

        Notes
        -------
        Each element of the `decomp_list` is a sample represented through a tensor factorisation. Each factor vector
        of each factor matrix of such decomposition is treated as a sample. Next we combine factor vectors from
        different factor matrices into a new sample by concatenating these factor vectors. Finally, we repeat this for
        each sample from the `decomp_list` and stack vertically the corresponding new samples.

        For each sample, the combination of vectors from several np.ndarrays is implemented as proposed in
        https://stackoverflow.com/a/47144986/6147064. This results in a matrix [n_new_samples, n_new_features].
        Next unfold this matrix into a row vector. Concatenated obtained rows for each sample from `decomp_list`.
        Finally, split the obtained array into a set of new datasets.
        """
        sample_decomp = decomp_list[0]

        # Original sample_decomp.fmat[i] takes form [n_features, n_vectors], n_vectors are treated as samples
        n_list = [fmat.T.shape[0] for fmat in sample_decomp.fmat]  # list of numbers of factor vectors (samples)
        m_list = [fmat.T.shape[1] for fmat in sample_decomp.fmat]  # list of numbers of features in each factor vector

        # new number of samples (all combinations of all factor vectors from different factor matrices)
        rows = np.asscalar(np.prod(np.array(n_list)))
        # new number of features (sum of features in each )
        cols = np.asscalar(np.sum(np.array(m_list)))

        arrays = [np.arange(n) for n in n_list]
        idx = np.meshgrid(*arrays, indexing='ij')

        A = sample_decomp.fmat[0].T  # Transpose is crucial
        B = sample_decomp.fmat[1].T  # Transpose is crucial
        C = sample_decomp.fmat[2].T  # Transpose is crucial
        temp = np.concatenate((A[idx[0], :], B[idx[1], :], C[idx[2], :]), axis=-1).reshape(rows, cols)
        data = temp.flatten()

        for sample in range(1, len(decomp_list)):
            sample_decomp = decomp_list[sample]
            A = sample_decomp.fmat[0].T  # Transpose is crucial
            B = sample_decomp.fmat[1].T  # Transpose is crucial
            C = sample_decomp.fmat[2].T  # Transpose is crucial
            temp = np.concatenate((A[idx[0], :], B[idx[1], :], C[idx[2], :]), axis=-1).reshape(rows, cols)
            data = np.vstack((data, temp.flatten()))

        split_idx = np.arange(0, rows*cols+1, cols)
        data_list = np.hsplit(data, split_idx[1:-1])

        return data_list
