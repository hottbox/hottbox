import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    """ Accuracy classification score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) labels.

    y_pred : np.ndarray
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``True``, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.

    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (best performance is 1), else returns
        the number of correctly classified samples.
    """
    score = np.sum(y_pred == y_true) / y_true.size
    return score