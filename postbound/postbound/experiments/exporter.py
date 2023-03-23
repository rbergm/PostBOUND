from __future__ import annotations

import json

import pandas as pd

from postbound import postbound as pb
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
