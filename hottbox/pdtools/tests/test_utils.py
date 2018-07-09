"""
Tests for the utils module from pandas tools
"""
import pytest
import numpy as np
import pandas as pd
from itertools import product
from ..utils import *


def test_pd_to_tensor():
    content = dict(
        country=['UK', 'RUS'],
        year=[2005, 2015, 2010],
        month=['Jan', 'Feb', 'Mar', 'Apr'],
        day=['Mon', 'Wed', 'Fri']
    )
    data = list(product(*content.values()))
    columns = list(content.keys())
    df = pd.DataFrame(data=data, columns=columns)
    df['population'] = np.arange(df.shape[0])
    df_mi = df.set_index(columns)

    shape_labels = df_mi.index.names
    true_shape = tuple([len(content[label]) for label in shape_labels])
    true_mode_names = columns

    tensor = pd_to_tensor(df=df_mi, keep_index=True)
    assert tensor.shape == true_shape
    assert tensor.mode_names == true_mode_names
    for mode in tensor.modes:
        name = mode.name
        assert mode.index == content[name]

    tensor = pd_to_tensor(df_mi, keep_index=False)
    assert all([mode.index is None for mode in tensor.modes])


def test_tensor_to_pd():
    content = dict(
        country=['UK', 'RUS'],
        year=[2005, 2015, 2010],
        month=['Jan', 'Feb', 'Mar', 'Apr'],
        day=['Mon', 'Wed', 'Fri']
    )
    data = list(product(*content.values()))
    columns = list(content.keys())
    multi_index = columns
    df_base = pd.DataFrame(data=data, columns=columns)
    values = np.arange(df_base.shape[0])

    #----- test for default column name
    value_column_name = "Values"
    df_data = pd.DataFrame(data=values, columns=[value_column_name])
    df = pd.concat([df_base, df_data], axis=1)
    df_mi = df.set_index(multi_index)
    tensor = pd_to_tensor(df_mi, keep_index=True)
    df_rec = tensor_to_pd(tensor=tensor)
    pd.testing.assert_frame_equal(df_mi, df_rec)

    #----- test for custom column name
    value_column_name = "population"
    df_data = pd.DataFrame(data=values, columns=[value_column_name])
    df = pd.concat([df_base, df_data], axis=1)
    df_mi = df.set_index(multi_index)
    tensor = pd_to_tensor(df_mi, keep_index=True)
    df_rec = tensor_to_pd(tensor=tensor, col_name=value_column_name)
    pd.testing.assert_frame_equal(df_mi, df_rec)

    #----- test for not keeping the index values
    # TODO: Don't like this implementation but it works
    dims = [len(content[key]) for key in columns]
    combos = [np.arange(i) for i in dims]
    data = list(product(*combos))
    df_base = pd.DataFrame(data=data, columns=columns)
    value_column_name = "population"
    df_data = pd.DataFrame(data=values, columns=[value_column_name])
    df = pd.concat([df_base, df_data], axis=1)
    df_mi = df.set_index(multi_index)
    tensor = pd_to_tensor(df_mi, keep_index=False)
    df_rec = tensor_to_pd(tensor=tensor, col_name=value_column_name)
    pd.testing.assert_frame_equal(df_mi, df_rec)

    # ----- tests that should FAILS
    with pytest.raises(TensorStateError):
        tensor.unfold(0, inplace=True)
        tensor_to_pd(tensor=tensor)
