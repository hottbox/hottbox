import numpy as np
import pandas as pd
from ..core.structures import Tensor
from ..errors import TensorStateError


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from hottbox.pdtools import pd_to_tensor
    >>> data = {'Year': [2005, 2005, 2005, 2005, 2010, 2010, 2010, 2010],
    ...         'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan', 'Jan', 'Feb', 'Feb'],
    ...         'Day': ['Mon', 'Wed', 'Mon', 'Wed', 'Mon', 'Wed', 'Mon', 'Wed'],
    ...         'Population': np.arange(8)
    ...         }
    >>> df = pd.DataFrame.from_dict(data)
    >>> df.set_index(["Year", "Month", "Day"], inplace=True)
    >>> print(df)
                        Population
        Year Month Day
        2005 Jan   Mon           0
                   Wed           1
             Feb   Mon           2
                   Wed           3
        2010 Jan   Mon           4
                   Wed           5
             Feb   Mon           6
                   Wed           7
    >>> tensor = pd_to_tensor(df)
    >>> print(tensor.data)
        [[[0 1]
          [2 3]]
         [[4 5]
          [6 7]]]
    >>> print(tensor)
        This tensor is of order 3 and consists of 8 elements.
        Sizes and names of its modes are (2, 2, 2) and ['Year', 'Month', 'Day'] respectively.
    >>> tensor.modes
        [Mode(name='Year', index=[2005, 2010]),
         Mode(name='Month', index=['Jan', 'Feb']),
         Mode(name='Day', index=['Mon', 'Wed'])]
    >>> tensor = pd_to_tensor(df, keep_index=False)
    >>> tensor.modes
        [Mode(name='Year', index=None),
         Mode(name='Month', index=None),
         Mode(name='Day', index=None)]
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
            mode_index = {i: index}
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

    Raises
    ------
    TensorStateError
        If ``tensor`` is not in normal state: ``tensor.in_normal_state is False``.

    Examples
    --------

    1) Conversion of a tensor with default meta information

    >>> import numpy as np
    >>> from hottbox.core import Tensor
    >>> from hottbox.pdtools import tensor_to_pd
    >>> data = np.arange(8).reshape(2, 2, 2)
    >>> tensor = Tensor(data)
    >>> print(tensor.data)
        [[[0 1]
          [2 3]]
         [[4 5]
          [6 7]]]
    >>> tensor.modes
        [Mode(name='mode-0', index=None),
         Mode(name='mode-1', index=None),
         Mode(name='mode-2', index=None)]
    >>> df = tensor_to_pd(tensor)
    >>> print(df)
                                  Values
        mode-0 mode-1 mode-2
        0      0      0            0
                      1            1
               1      0            2
                      1            3
        1      0      0            4
                      1            5
               1      0            6
                      1            7

    2) Conversion of a tensor with specified mode names

    >>> import numpy as np
    >>> from hottbox.core import Tensor
    >>> from hottbox.pdtools import tensor_to_pd
    >>> data = np.arange(8).reshape(2, 2, 2)
    >>> tensor = Tensor(data, mode_names=["Year", "Month", "Day"])
    >>> print(tensor.data)
        [[[0 1]
          [2 3]]
         [[4 5]
          [6 7]]]
    >>> tensor.modes
        [Mode(name='Year', index=None),
         Mode(name='Month', index=None),
         Mode(name='Day', index=None)]
    >>> df = tensor_to_pd(tensor)
    >>> print(df)
                            Values
        Year Month Day
        0    0     0         0
                   1         1
             1     0         2
                   1         3
        1    0     0         4
                   1         5
             1     0         6
                   1         7

    3) Conversion of a tensor with specified mode names and mode index

    >>> import numpy as np
    >>> from hottbox.core import Tensor
    >>> from hottbox.pdtools import tensor_to_pd
    >>> data = np.arange(8).reshape(2, 2, 2)
    >>> mode_index = {0: [2005, 2010],
    ...               1: ["Jan", "Feb"],
    ...               2: ["Mon", "Wed"],
    ...              }
    >>> tensor = Tensor(data, mode_names=["Year", "Month", "Day"])
    >>> tensor.set_mode_index(mode_index)
    >>> print(tensor.data)
        [[[0 1]
          [2 3]]
         [[4 5]
          [6 7]]]
    >>> tensor.modes
        [Mode(name='Year', index=[2005, 2010]),
         Mode(name='Month', index=['Jan', 'Feb']),
         Mode(name='Day', index=['Mon', 'Wed'])]
    >>> df = tensor_to_pd(tensor)
    >>> print(df)
                          Values
        Year Month Day
        2005 Jan   Mon       0
                   Wed       1
             Feb   Mon       2
                   Wed       3
        2010 Jan   Mon       4
                   Wed       5
             Feb   Mon       6
                   Wed       7
    >>> df = tensor_to_pd(tensor, col_name="Population")
    >>> print(df)
                            Population
        Year Month Day
        2005 Jan   Mon           0
                   Wed           1
             Feb   Mon           2
                   Wed           3
        2010 Jan   Mon           4
                   Wed           5
             Feb   Mon           6
                   Wed           7
    """
    if not tensor.in_normal_state:
        raise TensorStateError("`tensor` should be in normal state prior this conversion")

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
    # data = tensor.data.ravel()
    data = tensor.vectorise(inplace=False).data

    # Create dataframe
    if col_name is None:
        col_name = "Values"
    df = pd.DataFrame(data=data, index=index, columns=[col_name])
    return df
