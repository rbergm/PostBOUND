"""Utilities to work with Pandas data frames"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def as_df(
    data: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Generates a new Pandas `DataFrame` based on an array-of-structs style data input.

    Each dictionary corresponds to one row of the dataframe. All dictionaries have to consist of exactly the same key-value
    pairs. Each key becomes a column in the dataframe. The precise columns are inferred from the first dictionary in the
    collection. In this case, column values are derived directly from the keys.
    """
    if not data:
        return pd.DataFrame()

    data_template = data[0]
    df_container: dict[str, list[Any]] = {col: [] for col in data_template.keys()}
    for row in data:
        for key in df_container.keys():
            df_container[key].append(row[key])
    return pd.DataFrame(df_container)


def read_df(path: Path | str, **kwargs) -> pd.DataFrame:
    """Reads a Pandas `DataFrame` from a file, inferring the file format from the file extension.

    All additional arguments are passed directly to the respective Pandas read function.
    """
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path, **kwargs)
        case ".parquet":
            return pd.read_parquet(path, **kwargs)
        case ".xlsx" | ".xls":
            return pd.read_excel(path, **kwargs)
        case ".json":
            return pd.read_json(path, **kwargs)
        case ".hdf" | ".h5" | ".hdf5":
            return pd.read_hdf(path, **kwargs)
        case ".feather":
            return pd.read_feather(path, **kwargs)
        case ".orc":
            return pd.read_orc(path, **kwargs)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def write_df(
    df: pd.DataFrame, path: Path | str, *, index: bool = False, **kwargs
) -> None:
    """Writes a Pandas `DataFrame` to a file, inferring the file format from the file extension.

    All additional arguments are passed directly to the respective Pandas write function.
    """
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            df.to_csv(path, index=index, **kwargs)
        case ".parquet":
            df.to_parquet(path, **kwargs)
        case ".xlsx" | ".xls":
            df.to_excel(path, index=index, **kwargs)
        case ".json":
            df.to_json(path, index=index, **kwargs)
        case ".hdf" | ".h5" | ".hdf5":
            df.to_hdf(path, **kwargs)
        case ".feather":
            df.to_feather(path, **kwargs)
        case ".orc":
            df.to_orc(path, **kwargs)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
