import numpy as np
from ..core.structures import Tensor
from ..utils import sliceT, genToeplitzMatrix
import itertools

class basicTensor():
    """ Generates a dense or sparse tensor of any dimension and fills it accordingly
    
    Parameters
    ----------
    dim : int
        specifies the dimensions of the tensor
    distr (optional): string
        Specifies the random generation using a class of the numpy.random module
    distr_type (optional) : int
        Number of indices to not fix. 0 will be applied globally, 1 will apply to fibers, 2 to slices, etc.

    Returns
    -------
    tensor: Tensor
        Generated tensor according to the parameters specified
    """
    
    def __init__(self, dim, distr='uniform', distr_type=0):
        self.dim = dim
        self.distr_type = distr_type
        self.distr = distr
    
    def _predefined_distr(self, dim):
        distrlist = {'uniform':np.random.uniform(size=dim), 
                        'normal':np.random.normal(size=dim),
                        'triangular': np.random.triangular(-1, 0, 1, size=dim),
                        'standard-t': np.random.standard_t(10, size=dim),
                        'ones': np.ones(dim),
                        'zeros': np.zeros(dim)}
        if self.distr not in distrlist:
            raise NameError("The distribution {} is not an available one.\
             Please refer to the list of implementations: {}".format(self.distr, distrlist.keys()))
        return distrlist[self.distr]
    
    def dense(self, fxdind=None):
        """ Defines a dense Tensor
        
        Returns
        -------
        tensor : Tensor
        """

        # fxdind: fixed indices 
        if self.distr_type == 0:
            tensor = self._predefined_distr(self.dim)
        else:
            tensor = np.random.uniform(size=self.dim)
            print("not yet implemented")
        return Tensor(array=tensor)
    
    def sparse(self, fxdind=None, pct=0.05):
        """ Define a sparse Tensor filling approximately pct% of the dataset
        
        Returns
        -------
        tensor : Tensor
        """
        tensorsz = np.product(self.dim)
        if self.distr_type == 0:
            sz = int(tensorsz * 0.05)
            tensor = np.zeros(tensorsz)
            indx = np.random.randint(low=0, high=tensorsz, size=sz)
            tensor[indx] = self._predefined_distr(sz)
            tensor = tensor.reshape(self.dim)
        else:
            print("not yet implemented")
        
        return Tensor(array=tensor)
    
    def superdiagonal(self, axis1=0, axis2=1):
        if self.dim[1:] != self.dim[:-1]:
            print("Must have equal dimensions "\
                + "for a supersymmetric matrix")
        
        tensor = np.zeros(self.dim)
        dataset = self._predefined_distr(self.dim[0])
        np.fill_diagonal(tensor, dataset)
        return Tensor(array=tensor)

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
        high=1000

    matSz = np.asarray(shape)[modes]
    numMats = int(np.prod(shape[~modes]))
    if matSz[0] == matSz[1]:
        szMult = matSz[0]
    else: # minimal generation
        szMult = matSz[0]+matSz[1]
    
    toepVals = np.random.randint(low,high,size=(numMats,szMult))
    for i in range(numMats):
        if matSz[0] == matSz[1]:
            _r,_c = toepVals[i], None
        else:
            _c,_r = toepVals[i][:matSz[0]], toepVals[i][matSz[0]:]     
        toepMatrix = genToeplitzMatrix(r=_r, c=_c)
        matC.append(toepMatrix)
    return matC

def toeplitzTensor(shape, modes=[0,1], matC=None, random=False, lh=(None,None)):
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
    low,high = lh
    # Generate a list of Toeplitz matrices
    if random:
        matC = _toeplitzRandom(shape,modes,low,high)
    tensor = np.zeros(shape=shape)
    matC = np.asarray(matC)
        
    if len(shape) == 2:
        return genToeplitzMatrix(matC)
    # Fix all axis except modes
    availmodes = np.setdiff1d(np.arange(dim_req),modes)
    availsz = np.asarray(shape)[availmodes]
    all_combs = []
    
    for sz in availsz:
        all_combs.append(np.arange(sz))
    all_combs = list(itertools.product(*all_combs))
    
    tsz = np.product(availsz[1:])
    for currmode in range(availsz[0]):
        modecombs = all_combs[tsz*currmode:tsz*(currmode+1)]
        for i, m in enumerate(modecombs):
            sliceT(tensor,np.asarray(m),np.asarray(availmodes),overwrite=matC[i+currmode])
    return Tensor(array=tensor)
