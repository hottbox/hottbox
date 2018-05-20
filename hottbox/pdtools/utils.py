import numpy as np
import pandas as pd
from ..core.structures import Tensor


def pd_to_tensor(df, keep_index=True):
    """ Represent multi-index pandas dataframe as a tensor

    Parameters
    ----------
    df : pd.DataFrame
        Multi-index dataframe with only one column of data
    keep_index : bool
        Keep level values of dataframe multi-index

    Returns
    -------
    tensor : Tensor
    """
    # TODO: need to think what should we do when multi-index dataframe is composed of several columns

    # Reshape values into multi-dimensional array
    dims = tuple([len(level) for level in df.index.levels])
    data = df.as_matrix().reshape(dims)

    # Get mode names
    mode_names = df.index.names

    # Create tensor
    tensor = Tensor(array=data, mode_names=mode_names)

    # Set index for each tensor mode
    if keep_index:
        multi_index = df.index
        for i in range(len(dims)):
            level_index = multi_index.get_level_values(i)
            level_index_names = level_index.get_values()
            idx = np.unique(level_index_names, return_index=True)[1]
            index = [level_index_names[j] for j in sorted(idx)]
            mode_index = {i : index}
            tensor.set_mode_index(mode_index)
    return tensor


def tensor_to_pd(tensor, col_name=None):
    """ Represent tensor as a multi-index pandas dataframe

    Parameters
    ----------
    tensor : Tensor
        Tensor to be represented as a multi-index dataframe
    col_name : str
        Column label to use for resulting dataframe

    Returns
    -------
    df : pd.DataFrame
        Multi-index data frame
    """
    if not tensor.in_normal_state:
        raise TypeError("`tensor` should be in normal state prior this conversion")

    # Create multidimensional index
    names = tensor.mode_names
    all_indices = [None] * tensor.order
    for i, mode in enumerate(tensor.modes):
        if mode.index is None:
            all_indices[i] = [j for j in range(tensor.shape[i])]
        else:
            all_indices[i] = mode.index
    index = pd.MultiIndex.from_product(all_indices, names=names)

    # Vectorise values (!!! keep in mind, tensor should not be modified in anyway !!!)
    # data = tensor.unfold(mode=0, inplace=False).data.ravel()
    data = tensor.data.ravel()

    # Create dataframe
    if col_name is None:
        col_name = "Values"
    df = pd.DataFrame(data=data, index=index, columns=[col_name])
    return df
