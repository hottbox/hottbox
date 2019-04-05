"""
Helper functions for generating synthetic tensors
"""
import itertools
import numpy as np
from scipy.linalg import toeplitz as toeplitz_mat
from hottbox.utils.generation.basic import dense_tensor


def _toeplitz_random(shape, modes, low=None, high=None):
    """ Generate the apppropraite number of Toeplitz matrices
    according to the required shape

    Returns
    -------
    matC: list[np.ndarray]
        List of Toeplitz Matrices
    """
    modes = np.asarray(modes)
    shape = np.asarray(shape)
    matC = []
    if low is None:
        low = 0
    if high is None:
        high = 1000

    matSz = np.asarray(shape)[modes]
    numMats = int(np.prod(shape[~modes]))
    if matSz[0] == matSz[1]:
        szMult = matSz[0]
    else:  # minimal generation
        szMult = matSz[0]+matSz[1]

    toepVals = np.random.randint(low, high, size=(numMats, szMult))
    for i in range(numMats):
        if matSz[0] == matSz[1]:
            _r, _c = toepVals[i], toepVals[i].ravel().conjugate()
        else:
            _c, _r = toepVals[i][:matSz[0]], toepVals[i][matSz[0]:]
        toepMatrix = toeplitz_mat(r=_r, c=_c)
        matC.append(toepMatrix)
    return matC


def toeplitz_tensor(shape, modes=None, matC=None, random=False, lh=(None, None)):
    """ Function to generate a Toeplitz tensor. Every slice along modes will be a Toeplitz matrix.

    Parameters
    ----------
    shape : tuple(int)
        Shape of output. If matC is not None, they must match.
    modes : int or list(int)
        The mode by which the tensor is expected to be circulant
    matC : list(np.ndarray) or list(float)
        (optional) if None, random is set to True. Two input options.
    random : bool
        (optional) if true, input matC is ignored
    lh : tuple(float, float)
        (optional) used with random to define min and max values
    """
    dim_req = len(shape)
    if matC is None:
        random = True
    if modes is None:
        modes = [0, 1]
    if len(shape) == 1:
        raise ValueError("Toeplitz must have more than one dimension")
    low, high = lh
    # Generate a list of Toeplitz matrices
    if random:
        matC = _toeplitz_random(shape, modes, low, high)
    tensor = dense_tensor(shape, 'zeros')
    matC = np.asarray(matC)

    if len(shape) == 2:
        return toeplitz_mat(r=matC, c=matC.ravel().conjugate())
    # Fix all axis except modes
    availmodes = np.setdiff1d(np.arange(dim_req), modes)
    availsz = np.asarray(shape)[availmodes]
    all_combs = []

    for sz in availsz:
        all_combs.append(np.arange(sz))
    all_combs = list(itertools.product(*all_combs))

    tsz = np.product(availsz[1:])
    for currmode in range(availsz[0]):
        modecombs = all_combs[tsz*currmode:tsz*(currmode+1)]
        for i, m in enumerate(modecombs):
            tensor.write_subtensor(np.asarray(m), np.asarray(availmodes), matC[i+currmode])
    return tensor
