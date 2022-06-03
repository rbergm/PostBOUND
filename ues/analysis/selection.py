
from typing import Iterable, Union

import pandas as pd


def best_total_run(data: pd.DataFrame, group_cols: Union[Iterable[str], str], *,
                   run_col: str = "run", performance_col: str = "rt_total") -> pd.DataFrame:
    if not isinstance(group_cols, list) and not isinstance(group_cols, tuple):
        group_cols = [group_cols]

    total_performance = data.groupby(group_cols + run_col)[performance_col].sum()
    best_runs = total_performance.loc[total_performance.groupby].reset_index().drop(columns=performance_col)
    return data.merge(best_runs, how="inner", on=group_cols)


def best_query_repetition(data: pd.DataFrame, group_cols: Union[Iterable[str], str], *,
                          performance_col: str = "rt_total") -> pd.DataFrame:
    if not isinstance(group_cols, list) and not isinstance(group_cols, tuple):
        group_cols = [group_cols]

    best_repetition = data.groupby(group_cols)[performance_col].idxmin().to_frame().reset_index()
    return data.loc[best_repetition[performance_col]].reset_index(drop=True)
