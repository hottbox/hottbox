import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from .._temp_utils import get_meta, IMG_SIZE, load_as_original

from ....core.structures import Tensor
from ..stm import LSSTM


np.random.seed(0)


class TestLSSTM:

    def test_init(self):
        C, tol, max_iter = 1, 1, 1
        stm = LSSTM(C=C, tol=tol, max_iter=max_iter)
        assert stm.C == C
        assert stm.tol == tol
        assert stm.max_iter == max_iter
        assert stm.weights_ is None
        assert stm.bias_ is None
        assert stm.bias_history_ is None
        assert stm.eta_history_ is None
        assert stm.probability is False
        assert stm.verbose is False
        assert stm.name == "LSSTM"

    # def test_set_params(self):
    #     pass
    #
    # def test_get_params(self):
    #     pass
    #
    def test_fit(self):
        """ Sanity test for `fit` method """
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        labels_train = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]

        max_iter = 20
        stm = LSSTM(max_iter=max_iter)
        stm.fit(X=data_train, y=labels_train)

        assert len(stm.weights_) == len(sample_shape)

        assert isinstance(stm.bias_, float)
        assert stm.bias_history_.size >= 1
        assert stm.bias_history_.size <= max_iter

        assert stm.eta_history_.shape[0] <= max_iter
        assert stm.eta_history_.shape[0] >= 1
        assert stm.eta_history_.shape[1] <= len(sample_shape)

    def test_fit_fail(self):
        """ Test for incorrect input data to `fit` method """
        stm = LSSTM()
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        labels_train = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]

        with pytest.raises(TypeError):
            # Data should be passed as list
            wrong_data = np.zeros(sample_shape + (n_samples_train,))
            stm.fit(X=wrong_data, y=labels_train)

        with pytest.raises(TypeError):
            # Data should be a list of 'Tensor'
            wrong_data = [np.zeros(sample_shape) for _ in range(n_samples_train)]
            stm.fit(X=wrong_data, y=labels_train)

        with pytest.raises(ValueError):
            # All tensors should be of the same order
            data_array = np.ones(sample_shape)
            wrong_data = [Tensor(data_array) for _ in range(n_samples_train)]
            wrong_data[0] = Tensor(np.expand_dims(data_array, 0))
            stm.fit(X=wrong_data, y=labels_train)

        with pytest.raises(ValueError):
            # All tensors should be of the same shape
            data_array = np.ones(sample_shape)
            wrong_data = [Tensor(data_array) for _ in range(n_samples_train)]
            wrong_data[0] = Tensor(np.moveaxis(data_array, 0, -1))
            stm.fit(X=wrong_data, y=labels_train)

        with pytest.raises(ValueError):
            # Labels should form a binary set of numbers
            wrong_labels = np.arange(n_samples_train)
            stm.fit(X=data_train, y=wrong_labels)

        with pytest.raises(ValueError):
            # Number of labels should match a number of data samples
            wrong_labels = labels_train[:-1]
            stm.fit(X=data_train, y=wrong_labels)

        with pytest.raises(ValueError):
            # Number of data samples should match a number of labels
            wrong_data = data_train[:-1]
            stm.fit(X=wrong_data, y=labels_train)

    def test_predict(self):
        """ Sanity test for `predict` method """
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        n_samples_test = n_samples_train * 2
        labels_train = np.concatenate([np.ones(n_samples_1_train)*5, np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_test)]

        stm = LSSTM()
        stm.fit(X=data_train, y=labels_train)
        labels_predicted = stm.predict(X=data_test)
        assert isinstance(labels_predicted, np.ndarray)
        assert labels_predicted.size == len(data_test)
        assert np.unique(labels_predicted) in np.unique(labels_train)

    def test_predict_fail(self):
        """ Test for premature use or incorrect input data to `predict` method """
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        n_samples_test = n_samples_train * 2
        labels_train = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_test)]

        stm = LSSTM()
        with pytest.raises(ValueError):
            # Should call 'fit' beforehand
            stm.predict(data_test)

        stm.fit(X=data_train, y=labels_train)

        with pytest.raises(TypeError):
            # Data should be passed as list
            wrong_data = np.zeros(sample_shape + (n_samples_train,))
            stm.predict(X=wrong_data)

        with pytest.raises(TypeError):
            # Data should be passed as list of 'Tensor'
            wrong_data = [np.zeros(sample_shape) for _ in range(n_samples_train)]
            stm.predict(X=wrong_data)

        with pytest.raises(ValueError):
            # All tensors should be of the same order
            data_array = np.ones(sample_shape)
            wrong_data = [Tensor(data_array) for _ in range(n_samples_train)]
            wrong_data[0] = Tensor(np.expand_dims(data_array, 0))
            stm.predict(X=wrong_data)

        with pytest.raises(ValueError):
            # All tensors should be of the same shape
            data_array = np.ones(sample_shape)
            wrong_data = [Tensor(data_array) for _ in range(n_samples_train)]
            wrong_data[0] = Tensor(np.moveaxis(data_array, 0, -1))
            stm.predict(X=wrong_data)

        with pytest.raises(ValueError):
            # Check that all samples in 'X' are of the same order as during training.
            data_array = np.ones(sample_shape)
            wrong_data = [Tensor(np.expand_dims(data_array, -1)) for _ in range(n_samples_train)]
            stm.predict(wrong_data)

        with pytest.raises(ValueError):
            # Check that all samples in 'X' are of the same shape as during training.
            data_array = np.ones(sample_shape[::-1])
            wrong_data = [Tensor(data_array) for _ in range(n_samples_train)]
            stm.predict(wrong_data)


    # def test_predict_proba(self):
    #     stm = LSSTM()
    #     stm.predict_proba()
    #
    def test_score(self):
        # TODO: add check that test labels are within train labels. This also needs to be incorporated in assert method for test data
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        n_samples_test = n_samples_train * 2
        labels_train = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        labels_test = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_test)]

        stm = LSSTM()
        stm.fit(X=data_train, y=labels_train)
        # acc_score = stm.score(X=data_test,y=labels_test)

        # assert acc_score <= 1 and acc_score >= 0