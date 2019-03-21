"""
Tests for the base module of classification algorithms
"""
import pytest
from ..base import Classifier


class TestClassifier:
    """ Tests for the Classifier as an interface"""

    def test_init(self):
        probability = False
        verbose = False
        classifier_interface = Classifier(probability=probability,
                                          verbose=verbose
                                          )
        assert classifier_interface.name == "Classifier"
        assert classifier_interface.probability == probability
        assert classifier_interface.verbose == verbose

        with pytest.raises(NotImplementedError):
            classifier_interface.fit(X=None, y=None)
        with pytest.raises(NotImplementedError):
            classifier_interface.predict(X=None)
        with pytest.raises(NotImplementedError):
            classifier_interface.predict_proba(X=None)
        with pytest.raises(NotImplementedError):
            classifier_interface.score(X=None, y=None)

    def test_get_set_params(self):
        probability = False
        verbose = False
        classifier_interface = Classifier(probability=probability,
                                          verbose=verbose
                                          )
        params = classifier_interface.get_params()
        assert len(params.keys()) == 2
        assert params['probability'] == probability
        assert params['verbose'] == verbose

        # Test for empty use of set_params method
        classifier_interface.set_params()

        classifier_interface.set_params(probability=(not probability),
                                        verbose=(not verbose)
                                        )
        params = classifier_interface.get_params()
        assert params['probability'] == (not probability)
        assert params['verbose'] == (not verbose)

        # Test for setting an invalid parameter
        with pytest.raises(ValueError):
            classifier_interface.set_params(dummy_param=True)


