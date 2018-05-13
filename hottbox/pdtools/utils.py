import numpy as np
import pandas as pd
from itertools import product
from functools import reduce
from collections import OrderedDict
from ..core.structures import Tensor


def pd_to_tensor(df, time_col=None):
    """ Represent a multi-index pandas dataframe as a tensor

    Parameters
    ----------
    df : pd.DataFrame
        Multi-index dataframe with only one column of data
    time_col : str

    Returns
    -------
    tensor : Tensor
    """
    # TODO: clean this mess ))
    # time_col (if declared) is the name of the index column associated to time

    # assume all index columns are data-modes
    mode_cols = list(df.index.names)
    mode_names = OrderedDict([(i, name) for i, name in enumerate(mode_cols)])

    # if time mode declared, reorder the index columns such that time is the last mode
    if time_col is not None:
        mode_cols.append(mode_cols.pop(mode_cols.index(time_col)))  # bring time column to end of list
        time_vals = np.unique(df.index.get_level_values(time_col))  # get unique time values

    # get dimensionality of each mode
    dims = []
    for mode in mode_cols: dims.append(np.unique(df.index.get_level_values(mode)).shape[0])

    # reorder data into associated tensor format
    data = df.as_matrix().reshape(*dims)
    tensor = Tensor(array=data, mode_names=mode_names)

    return tensor


def tensor_to_pd(tensor, time_mode=False):
    """ Represent a tensor as a multi-index pandas dataframe

    Parameters
    ----------
    tensor : Tensor
        Tensor to be represented as a multi-index dataframe
    time_mode : int

    Returns
    -------
    df : pd.DataFrame
        Multi-index data frame
    """
    dims = tensor.shape

    # for each mode, introduce dummy variable names (integers)
    combos = [np.arange(i) for i in dims]

    # compute all tensor element indices based on mode-wise variable names
    prod_combos = np.array(
        list(product(*combos))
    )

    df = pd.DataFrame()  # create empty panda
    # df['Value'] = tensor.unfold(mode=(tensor.order-1), inplace=False).data.ravel()
    # df['Value'] = tensor.unfold(mode=-1, inplace=False).data.ravel()
    df['Value'] = tensor.unfold(mode=0, inplace=False).data.ravel()
    for i in np.arange(len(dims)):
        df[tensor.mode_names[i]] = prod_combos[:, i]

    df = df.set_index(list(df.columns[1:]))

    if time_mode is not False:
        ix_names = list(df.index.names)
        ix_names[time_mode] = 'Time'
        df.index.names = ix_names

    return df