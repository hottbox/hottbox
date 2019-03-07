import numpy as np
from ...core.structures import Tensor
from ...utils import sliceT, genToeplitzMatrix
import itertools

def _toeplitzRandom(shape, modes, low=None, high=None):
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
            _r, _c = toepVals[i], None
        else:
            _c, _r = toepVals[i][:matSz[0]], toepVals[i][matSz[0]:]
        toepMatrix = genToeplitzMatrix(r=_r, c=_c)
        matC.append(toepMatrix)
    return matC


def toeplitzTensor(shape, modes=[0, 1], matC=None, random=False, lh=(None, None)):
    """ Function to generate a toeplitz tensor. Every slice along modes will be a Toeplitz matrix.
    :param shape: (required) shape of output. If c is not None, they must match.
    :param modes: the mode by which the tensor is excpected to be circulant
    :param c: (optional) if None, random is set to True. Two input options.
                - A list of toeplitz matrices - assumed
                - A list of numbers
    :param random: (optional) if true, input c is ignored
    :param (low,high): (optional) used with random to define min and max values
    """
    dim_req = len(shape)
    if matC is None:
        random = True

    if len(shape) == 1:
        raise ValueError("Toeplitz must have more than one dimension")
    low, high = lh
    # Generate a list of Toeplitz matrices
    if random:
        matC = _toeplitzRandom(shape, modes, low, high)
    tensor = np.zeros(shape=shape)
    matC = np.asarray(matC)

    if len(shape) == 2:
        return genToeplitzMatrix(matC)
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
            sliceT(tensor, np.asarray(m), np.asarray(
                availmodes), overwrite=matC[i+currmode])
    return Tensor(array=tensor)
