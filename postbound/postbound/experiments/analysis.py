"""Provides utilities to work with benchmark results, including analysis and data preparation."""

from __future__ import annotations

import json

import natsort
import numpy as np
import pandas as pd

from postbound.experiments import runner
from postbound.util import jsonize


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms all non-scalar columns of the result data frame to enable an easy CSV export.

    This function handles two main aspects: 1) making sure that the query result can be written to CSV and 2) making
    sure that the description of the optimization pipeline can be written to CSV. In both cases, the column values will
    be transformed to JSON-objects if necessary.
    """
    if not len(df):
        return df

    prepared_df = df.copy()

    example_result = prepared_df[runner.COL_RESULT].iloc[0]
    if isinstance(example_result, list) or isinstance(example_result, tuple) or isinstance(example_result, dict):
        prepared_df[runner.COL_RESULT] = prepared_df[runner.COL_RESULT].apply(json.dumps)

    if runner.COL_OPT_SETTINGS in prepared_df:
        prepared_df[runner.COL_OPT_SETTINGS] = prepared_df[runner.COL_OPT_SETTINGS].apply(jsonize.to_json)

    return prepared_df


def sort_results(results_df: pd.DataFrame,
                 by_column: str | tuple[str] = (runner.COL_LABEL, runner.COL_EXEC_IDX)) -> pd.DataFrame:
    """Sorts the rows of the result data naturally.

    In contrast to lexicographic sorting, natural sorting handles numeric labels in a better way: labels like
    1a, 10a and 100a are sorted in this order instead of in reverse.

    The primary column for the sorting is the `label_column` which contains the actual labels. The secondary
    `execution_index_column` is used to break ties (and sorted lexicographically as well).
    """
    return results_df.sort_values(by=by_column,
                                  key=lambda series: np.argsort(natsort.index_natsorted(series)))
