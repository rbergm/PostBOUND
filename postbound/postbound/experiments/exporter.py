"""Provides utilities to work with benchmark results, including analysis and data preparation."""

from __future__ import annotations

import json

import natsort
import numpy as np
import pandas as pd

from postbound.experiments import runner


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return df

    prepared_df = df.copy()

    example_result = prepared_df[runner.COL_RESULT].iloc[0]
    if isinstance(example_result, list) or isinstance(example_result, tuple) or isinstance(example_result, dict):
        prepared_df[runner.COL_RESULT] = prepared_df[runner.COL_RESULT].apply(json.dumps)

    if runner.COL_OPT_SETTINGS in prepared_df:
        prepared_df[runner.COL_OPT_SETTINGS] = prepared_df[runner.COL_OPT_SETTINGS].apply(json.dumps)

    return prepared_df


def sort_results(results_df: pd.DataFrame, by_column: str = "label") -> pd.DataFrame:
    return results_df.sort_values(by=by_column,
                                  key=lambda _: np.argsort(natsort.index_natsorted(results_df[by_column])))
