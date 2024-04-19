"""Provides utilities to work with benchmark results, including some analysis tools and data export functions."""

from __future__ import annotations

import json
from typing import Optional

import natsort
import numpy as np
import pandas as pd

from postbound.db import db
from postbound.qal import qal
from postbound.experiments import runner
from postbound.optimizer import jointree
from postbound.util import jsonize, stats as num


def prepare_export(results_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies a benchmark result dataframe such that it can be written to CSV files without problems.

    This mostly involves converting Python objects to JSON counterparts that allow a reconstruction of equivalent data.

    More specifically, the function handles two main aspects:

    1. making sure that the query result can be written to CSV, and
    2. making sure that the description of the optimization pipeline can be written to CSV.

    In both cases, the column values will be transformed to JSON-objects if necessary.

    Parameters
    ----------
    results_df : pd.DataFrame
        The result dataframe created by one of the benchmark functions

    Returns
    -------
    pd.DataFrame
        The prepared dataframe

    See Also
    --------
    postbound.experiments.runner : Functions to obtain benchmark results
    """
    if not len(results_df):
        return results_df

    prepared_df = results_df.copy()

    example_result = prepared_df[runner.COL_RESULT].iloc[0]
    if isinstance(example_result, list) or isinstance(example_result, tuple) or isinstance(example_result, dict):
        prepared_df[runner.COL_RESULT] = prepared_df[runner.COL_RESULT].apply(json.dumps)

    if runner.COL_OPT_SETTINGS in prepared_df:
        prepared_df[runner.COL_OPT_SETTINGS] = prepared_df[runner.COL_OPT_SETTINGS].apply(jsonize.to_json)
    if runner.COL_DB_CONFIG in prepared_df:
        prepared_df[runner.COL_DB_CONFIG] = prepared_df[runner.COL_DB_CONFIG].apply(jsonize.to_json)

    return prepared_df


def sort_results(results_df: pd.DataFrame,
                 by_column: str | tuple[str] = (runner.COL_LABEL, runner.COL_EXEC_IDX)) -> pd.DataFrame:
    """Provides a better sorting of the benchmark results in a data frame.

    By default, the entries in the result data frame will be sorted either sequentially, or by a lexicographic ordering on the
    label column. This function uses a natural ordering over the column labels.

    In contrast to lexicographic sorting, natural sorting handles numeric labels in a better way: labels like
    1a, 10a and 100a are sorted in this order instead of in reverse.

    Parameters
    ----------
    results_df : pd.DataFrame
        Data frame containing the results to sort
    by_column : str | tuple[str], optional
        The columns by which to order, by default (runner.COL_LABEL, runner.COL_EXEC_IDX). A lexicographic ordering will
        be applied to all of them.

    Returns
    -------
    pd.DataFrame
        A reordered data frame. The original data frame is not modified
    """
    return results_df.sort_values(by=by_column,
                                  key=lambda series: np.argsort(natsort.index_natsorted(series)))


def possible_plans_bound(query: qal.SqlQuery, *,
                         join_operators: set[str] = {"nested-loop join", "hash join", "sort-merge join"},
                         scan_operators: set[str] = {"sequential scan", "index scan"}) -> int:
    """Computes a quick upper bound on the maximum number of possible query execution plans for a given query.

    This upper bound is a very coarse one, based on three assumptions:

    1. any join sequence (even involving cross-products) of any form (i.e. right-deep, bushy, ...) is allowed
    2. the choice of scan operators and join operators can be varied freely
    3. each table can be scanned using arbitrary operators

    The number of real-world query execution plans will typically be much smaller, because cross-products are only
    used if really necessary and the selected join operator influences the scan operators and vice-versa.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the bound should be computed
    join_operators : set[str], optional
        The allowed join operators, by default {"nested-loop join", "hash join", "sort-merge join"}
    scan_operators : set[str], optional
        The allowed scan operators, by default {"sequential scan", "index scan"}

    Returns
    -------
    int
        An upper bound on the number of possible query execution plans
    """
    n_tables = len(query.tables())

    join_orders = num.catalan_number(n_tables)
    joins = (n_tables - 1) * len(join_operators)
    scans = n_tables * len(scan_operators)

    return join_orders * joins * scans


def actual_plan_cost(query: qal.SqlQuery, analyze_plan: db.QueryExecutionPlan, *,
                     database: Optional[db.Database] = None) -> float:
    """Utility to compute the true cost of a query plan based on the actual cardinalities.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to analyze
    analyze_plan : db.QueryExecutionPlan
        The executed query which also contains the true cardinalities
    database : Optional[db.Database], optional
        The database providing the cost model. If omitted, the database is inferred from the database pool.

    Returns
    -------
    float
        _description_
    """
    database = database if database is not None else db.DatabasePool().get_instance()
    physical_plan = jointree.PhysicalQueryPlan.load_from_query_plan(analyze_plan, query)
    hinted_query = database.hinting().generate_hints(query, physical_plan)
    return database.optimizer().cost_estimate(hinted_query)


def text_diff(left: str, right: str, *, sep: str = " | ") -> str:
    """Merges two text snippets to allow for a comparison on a per-line basis.

    The two snippets are split into their individual lines and then merged back together.

    Parameters
    ----------
    left : str
        The text snippet to display on the left-hand side.
    right : str
        The text snippet to display on the right-hand side.
    sep : str, optional
        The separator to use between the left and right text snippets, by default `` | ``.

    Returns
    -------
    str
        The combined text snippet
    """
    left_lines = left.splitlines()
    right_lines = right.splitlines()

    max_left_len = max(len(line) for line in left_lines)
    left_lines_padded = [line.ljust(max_left_len) for line in left_lines]

    merged_lines = [f"{left_line}{sep}{right_line}" for left_line, right_line in zip(left_lines_padded, right_lines)]
    return "\n".join(merged_lines)
