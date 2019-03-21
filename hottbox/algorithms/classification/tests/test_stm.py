import pytest
import numpy as np
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
        assert stm.probability == False
        assert stm.verbose == False
        assert stm.weights_ is None
        assert stm.bias_ is None
        assert stm.bias_history_ is None
        assert stm.eta_history_ is None
        assert stm.probability is False
        assert stm.verbose is False
        assert stm.name == "LSSTM"

    def test_get_set_params(self):
        orig_params = dict(C=1,
                           tol=1,
                           max_iter=1,
                           probability=True,
                           verbose=True
                           )
        new_params = dict(C=10,
                          tol=10,
                          max_iter=10,
                          probability=False,
                          verbose=False
                          )
        stm = LSSTM(**orig_params)


        params = stm.get_params()
        all_fields = stm.__dict__
        for key in all_fields.keys():
            if key not in params.keys():
                # Test for wrong use of `set_params` (e.g. trying to modify attributes)
                with pytest.raises(ValueError):
                    wrong_params = {key: None}
                    stm.set_params(**wrong_params)
            else:
                # Test for `get_params`
                assert params[key] == orig_params[key]

        # Test for correct use of `set_params`
        stm.set_params(**new_params)
        params = stm.get_params()
        for key in params.keys():
            assert params[key] == new_params[key]



    @pytest.mark.parametrize("label_0, label_1", [
        (6, 8),
        ("word_1", "word_2"),
        ("word 1 with space", "word 2 with space")
    ])
    def test_fit(self, label_0, label_1):
        """ Sanity test for `fit` method """
        sample_shape = (2, 3, 4)
        n_samples_0_train = 5
        n_samples_1_train = 5

        labels_0_train = [label_0] * n_samples_0_train
        labels_1_train = [label_1] * n_samples_1_train
        labels_train = np.concatenate([np.array(labels_0_train), np.array(labels_1_train)])

        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_train.size)]

        max_iter = 20
        stm = LSSTM(max_iter=max_iter)
        stm.fit(X=data_train, y=labels_train)

        assert len(stm.weights_) == len(sample_shape)
        assert isinstance(stm.bias_, float)
        assert stm.bias_history_.size >= 1
        assert stm.bias_history_.size <= max_iter
        assert stm.eta_history_.shape[0] >= 1
        assert stm.eta_history_.shape[0] <= max_iter
        assert stm.eta_history_.shape[1] <= len(sample_shape)

    def test_fit_fail(self):
        """ Test for incorrect input data to `fit` method """
        stm = LSSTM()
        sample_shape = (2, 3, 4)
        n_samples_0_train = 5
        n_samples_1_train = 5
        n_samples_train = n_samples_0_train + n_samples_1_train
        labels_train = np.concatenate([np.ones(n_samples_0_train), np.zeros(n_samples_1_train)])
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

    @pytest.mark.parametrize("label_0, label_1", [
        (6, 8),
        ("word_1", "word_2"),
        ("word 1 with space", "word 2 with space")
    ])
    def test_predict(self, label_0, label_1):
        """ Sanity test for `predict` method """
        sample_shape = (2, 3, 4)
        n_samples_0_train = 5
        n_samples_1_train = 5

        labels_0_train = [label_0] * n_samples_0_train
        labels_1_train = [label_1] * n_samples_1_train
        labels_train = np.concatenate([np.array(labels_0_train), np.array(labels_1_train)])

        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_train.size)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_train.size * 2)]

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
        n_samples_test = n_samples_train
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

    def test_predict_proba(self):
        """ Sanity test for `predict_proba` method"""
        sample_shape = (2, 3, 4)
        n_samples_1_train = 5
        n_samples_2_train = 5
        n_samples_train = n_samples_1_train + n_samples_2_train
        n_samples_test = n_samples_train
        labels_train = np.concatenate([np.ones(n_samples_1_train), np.zeros(n_samples_2_train)])
        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_train)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(n_samples_test)]

        stm = LSSTM()
        with pytest.raises(NotImplementedError):
            stm.fit(X=data_train, y=labels_train)
            stm.predict_proba(X=data_test)

    @pytest.mark.parametrize("label_0, label_1", [
        (6, 8),
        ("word_1", "word_2"),
        ("word 1 with space", "word 2 with space")
    ])
    def test_score(self, label_0, label_1):
        """ Sanity test for `score` method """
        sample_shape = (2, 3, 4)
        n_samples_0_train = 5
        n_samples_1_train = 5
        n_samples_0_test = 2
        n_samples_1_test = 2

        labels_0_train = [label_0] * n_samples_0_train
        labels_1_train = [label_1] * n_samples_1_train
        labels_train = np.concatenate([np.array(labels_0_train), np.array(labels_1_train)])

        labels_0_test = [label_0] * n_samples_0_test
        labels_1_test = [label_1] * n_samples_1_test
        labels_test = np.concatenate([np.array(labels_0_test), np.array(labels_1_test)])

        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_train.size)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_test.size)]

        stm = LSSTM()
        stm.fit(X=data_train, y=labels_train)
        acc_score = stm.score(X=data_test,y=labels_test)
        assert acc_score <= 1 and acc_score >= 0

    @pytest.mark.parametrize("label_0, label_1", [
        (6, 8),
        # ("word_1", "word_2"),
        # ("word 1 with space", "word 2 with space")
    ])
    def test_score_fail(self, label_0, label_1):
        sample_shape = (2, 3, 4)
        n_samples_0_train = 5
        n_samples_1_train = 5
        n_samples_0_test = 2
        n_samples_1_test = 2

        labels_0_train = [label_0] * n_samples_0_train
        labels_1_train = [label_1] * n_samples_1_train
        labels_train = np.concatenate([np.array(labels_0_train), np.array(labels_1_train)])

        labels_0_test = [label_0 + 1] * n_samples_0_test
        labels_1_test = [label_1 + 1] * n_samples_1_test
        labels_test = np.concatenate([np.array(labels_0_test), np.array(labels_1_test)])

        data_train = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_train.size)]
        data_test = [Tensor(np.random.randn(*sample_shape)) for _ in range(labels_test.size)]

        stm = LSSTM()
        stm.fit(X=data_train, y=labels_train)
        with pytest.raises(ValueError):
            stm.score(X=data_test,y=labels_test)
