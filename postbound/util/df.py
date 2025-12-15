"""Utilities to work with Pandas data frames"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Optional

import pandas as pd


def _df_from_dict(
    data: dict[Any, Collection[Any]],
    key_name: Optional[str] = None,
    column_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    data_template = next(iter(data.values()))
    if column_names is None:
        column_name_map = {i: str(i) for i in range(len(data_template))}
    else:
        column_name_map = {idx: col for idx, col in enumerate(column_names)}

    df_container: dict[str, list[Any]] = {col: [] for col in column_name_map.values()}
    for row in data.values():
        for col_idx, col in enumerate(row):
            col_name = column_name_map[col_idx]
            df_container[col_name].append(col)

    key_name = "key" if key_name is None else key_name
    df_container[key_name] = list(data.keys())

    return pd.DataFrame(df_container)


def _df_from_list(data: Collection[dict[Any, Any]]) -> pd.DataFrame:
    data_template = next(iter(data))
    df_container: dict[str, list[Any]] = {col: [] for col in data_template.keys()}
    for row in data:
        for key in df_container.keys():
            df_container[key].append(row[key])
    return pd.DataFrame(df_container)


def as_df(
    data: dict[Any, Collection[Any]] | Collection[dict[Any, Any]],
    *,
    key_name: Optional[str] = None,
    column_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Generates a new Pandas `DataFrame`.

    The contents of the dataframe can be supplied in one of two forms: a collection of dictionaries will be transformed
    into a dataframe such that each dictionary corresponds to one row of the dataframe. All dictionaries have to
    consist of exactly the same key-value pairs. Each key becomes a column in the dataframe. The precise columns are
    inferred from the first dictionary in the collection. In this case, column values are derived directly from the
    keys.

    The other form consists of one large dictionary of keys mapping to several columns. The resulting dataframe will
    have one column that corresponds to the key values and additional columns that correspond to the entries in the
    collection which was mapped-to by the key. All collections have to consist of exactly the same number of elements.
    The precise number is inferred based on the first key-value pair. To name the different columns of the dataframe,
    the `key_name` and `column_names` can be used. If no key name is given, it defaults to `key`. If no column names
    are given, they default to numerical indices that correspond to the position in the mapped collection.
    """
    if not data:
        return pd.DataFrame()
    if isinstance(data, dict):
        return _df_from_dict(data, key_name, column_names)
    elif isinstance(data, Collection):
        return _df_from_list(data)
    else:
        raise TypeError("Unexpected data type: " + str(data))
