"""Utilities to optimize and execute queries and workloads in a reproducible and transparent manner."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from postbound import postbound as pb
from postbound.db import db
from postbound.qal import qal, transform, clauses
from postbound.optimizer import validation
from postbound.experiments import workloads

COL_LABEL = "label"
COL_QUERY = "query"
COL_QUERY_HINTS = "query_hints"

COL_T_EXEC = "exec_time"
COL_T_OPT = "optimization_time"
COL_RESULT = "query_result"

COL_EXEC_IDX = "execution_index"
COL_REP = "query_repetition"
COL_WORKLOAD_ITER = "workload_iteration"

COL_ORIG_QUERY = "original_query"
COL_OPT_SETTINGS = "optimization_settings"
COL_OPT_SUCCESS = "optimization_success"
COL_OPT_FAILURE_REASON = "optimization_failure_reason"

COL_DB_CONFIG = "db_config"


@dataclass
class ExecutionResult:
    """Captures all relevant components of a query optimization and execution result."""

    query: qal.SqlQuery
    """The query that was executed. If the query was optimized and transformed, these modifications are included."""

    result_set: object = None
    """The raw result set of the query, if it was executed"""

    optimization_time: float = np.nan
    """The time in seconds it took to optimized the query by PostBOUND.

    This does not account for optimization by the actual database system.
    """

    execution_time: float = np.nan
    """The time in seconds it took to execute the (potentially optimized) query by the actual database system.

    This execution time includes the entire end-to-end processing, i.e. starting with supplying the query to the
    database until the last byte of the result set was transferred back to PostBOUND. Therefore, this duration also
    includes the optimization time by the database system, as well as the entire time for data transfer.
    """


class QueryPreparationService:
    """This service handles transformations of input queries that are executed before running the query.

    These transformations mostly ensure that all queries in a workload provide the same type of result even in face
    of input queries that are structured slightly differently. For example, the preparation service can transform
    all the queries to be executed as `EXPLAIN` or `COUNT(*)` queries. Furthermore, the preparation service can
    store SQL statements that have to be executed before running the query. For example, a statement that disables
    parallel execution could be supplied here.
    """

    def __init__(self, *, explain: bool = False, count_star: bool = False, analyze: bool = False,
                 preparatory_statements: Optional[list[str]] = None):
        self.explain = explain
        self.analyze = analyze
        self.count_star = count_star
        self.preparatory_stmts = preparatory_statements if preparatory_statements else []

    def prepare_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        """Applies the selected transformations to the given input query."""
        if self.analyze:
            query = transform.as_explain(query, clauses.Explain.explain_analyze())
        elif self.explain:
            query = transform.as_explain(query, clauses.Explain.plan())

        if self.count_star:
            query = transform.as_count_star_query(query)

        return query

    def preparatory_statements(self) -> list[str]:
        """Provides all preparatory SQL statements that should be executed before running the query."""
        return self.preparatory_stmts


def _failed_execution_result(query: qal.SqlQuery, database: db.Database, repetitions: int = 1) -> pd.DataFrame:
    """Constructs a simple data frame that indicates a failed query execution."""
    return pd.DataFrame({
        COL_QUERY: [transform.drop_hints(query)] * repetitions,
        COL_QUERY_HINTS: [query.hints] * repetitions,
        COL_T_EXEC: [np.nan] * repetitions,
        COL_RESULT: [np.nan] * repetitions,
        COL_REP: list(range(1, repetitions + 1)),
        COL_DB_CONFIG: [database.describe()] * repetitions
    })


def _invoke_post_process(execution_result: ExecutionResult,
                         action: Optional[Callable[[ExecutionResult], None]] = None) -> None:
    """Executes the given post-process action if one has been supplied."""
    if not action:
        return
    action(execution_result)


def execute_query(query: qal.SqlQuery, database: db.Database, *,
                  repetitions: int = 1, query_preparation: Optional[QueryPreparationService] = None,
                  post_process: Optional[Callable[[ExecutionResult], None]] = None,
                  _optimization_time: float = np.nan) -> pd.DataFrame:
    """Runs the given query on the provided database.

    The query execution will be repeated for a total of `repetitions` times. Before the first repetition, the
    preparation service will be invoked if it has been supplied.

    In addition to the query execution, this function also accepts a `post_process` parameter. This parameter
    is a callable, that will be executed after each query run to perform arbitrary actions (e.g. online training of
    some models). The callable receives an `ExecutionResult` object as input and its output will not be handled in any
    way.

    The resulting data frame will be structured as follows:

    - the data frame will contain one row per query repetition
    - the current repetition is indicated in the `COL_REP` column
    - the actual query (without hints) is provided in the `COL_QUERY` column
    - the query hints are contained in the `COL_QUERY_HINTS` column
    - the execution time (end-to-end, i.e. until the last byte of the result set has been transferred back to
      PostBOUND) is provided in the `COL_T_EXEC` column (in seconds)
    - the query result of each repetition is contained in the `COL_RESULT` column
    """
    original_query = query
    if query_preparation:
        query = query_preparation.prepare_query(query)
        for stmt in query_preparation.preparatory_statements():
            database.cursor().execute(stmt)

    query_results = []
    execution_times = []
    for __ in range(repetitions):
        start_time = datetime.now()
        current_result = database.execute_query(query, cache_enabled=False)
        end_time = datetime.now()
        exec_time = (end_time - start_time).total_seconds()

        query_results.append(current_result)
        execution_times.append(exec_time)
        execution_result = ExecutionResult(query, current_result, _optimization_time, exec_time)
        _invoke_post_process(execution_result, post_process)

    return pd.DataFrame({
        COL_QUERY: [transform.drop_hints(original_query)] * repetitions,
        COL_QUERY_HINTS: [original_query.hints] * repetitions,
        COL_T_EXEC: execution_times,
        COL_RESULT: query_results,
        COL_REP: list(range(1, repetitions + 1)),
        COL_DB_CONFIG: [database.describe()] * repetitions
    })


def _wrap_workload(queries: Iterable[qal.SqlQuery] | workloads.Workload) -> workloads.Workload:
    """Transforms the given iterable of queries into a proper workload object, if it is not a workload already."""
    return queries if isinstance(queries, workloads.Workload) else workloads.generate_workload(queries)


def execute_workload(queries: Iterable[qal.SqlQuery] | workloads.Workload, database: db.Database, *,
                     workload_repetitions: int = 1, per_query_repetitions: int = 1, shuffled: bool = False,
                     query_preparation: Optional[QueryPreparationService] = None, include_labels: bool = False,
                     post_process: Optional[Callable[[ExecutionResult], None]] = None) -> pd.DataFrame:
    """Executes all the given queries on the provided database.

    Most of this process delegates to the `execute_query` method, so refer to its documentation for more details.
    The following are additional parameters for the workload execution:

    The entire workload will be repeated for an entire of `workload_repetitions` times. If `shuffled` is `True`,
    the order in which the queries are executed will be randomized _before_ each workload iteration. Note that this
    also applies for just a single repetition. The `include_labels` parameter expands the resulting data frame with
    another column that contains the query labels as obtained from the workload (integers are used if queries are
    supplied as an iterable rather than a workload object).

    In addition to the columns produced by `execute_query`, the resulting data frame will have the following columns:

    - an absolute index indicating which query is being executed (the first query has index 1, the query executed
      after that has index 2 and so on). This index will be increased for each query and workload repetition. I.e.,
      if a workload contains two queries, each query should be repeated 3 times and the entire workload should be
      repeated 4 times, the indexes will be 1..24. This information is contained in the `COL_EXEC_INDEX` column
    - the `COL_LABEL` column contains the query label if requested
    - the current workload iteration is described in the `COL_WORKLOAD_ITER` column
    """
    queries = _wrap_workload(queries)
    results = []
    current_execution_index = 1
    for i in range(workload_repetitions):
        current_repetition_results = []
        if shuffled:
            queries = queries.shuffle()

        for label, query in queries.entries():
            execution_result = execute_query(query, database, repetitions=per_query_repetitions,
                                             query_preparation=query_preparation, post_process=post_process)
            execution_result[COL_EXEC_IDX] = list(range(current_execution_index,
                                                        current_execution_index + per_query_repetitions))
            if include_labels:
                execution_result[COL_LABEL] = label

            current_repetition_results.append(execution_result)
            current_execution_index += per_query_repetitions

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

    result_df = pd.concat(results)
    target_labels = [COL_LABEL] if include_labels else []
    target_labels += [COL_EXEC_IDX, COL_QUERY,
                      COL_WORKLOAD_ITER, COL_REP,
                      COL_T_EXEC, COL_RESULT]
    return result_df[target_labels]


def optimize_and_execute_query(query: qal.SqlQuery, optimization_pipeline: pb.OptimizationPipeline, *,
                               repetitions: int = 1,
                               query_preparation: Optional[QueryPreparationService] = None,
                               post_process: Optional[Callable[[ExecutionResult], None]] = None) -> pd.DataFrame:
    """Optimizes the given query according to the settings of the `optimization_pipeline` and executes it afterwards.

    This function delegates most of its work to `execute_query`. In addition, the resulting data frame contains the
    following columns:

    - the time it took PostBOUND to optimize the query is indicated in the `COL_T_OPT` column (in seconds)
    - whether the query was optimized successfully is indicated in the `COL_OPT_SUCCESS` column
    - if the optimization failed, the reason(s) for the failure are contained in the `COL_OPT_FAILURE_REASON` column
    - the original query (i.e. before optimization and query preparation) is provided in the `COL_ORIG_QUERY` column
    - the selected optimization settings of the optimization pipeline is contained in the `COL_OPT_SETTINGS` column
    """
    try:
        start_time = datetime.now()
        optimized_query = optimization_pipeline.optimize_query(query)
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()

        execution_result = execute_query(optimized_query, repetitions=repetitions, query_preparation=query_preparation,
                                         database=optimization_pipeline.target_database(),
                                         post_process=post_process, _optimization_time=optimization_time)
        execution_result[COL_T_OPT] = optimization_time
        execution_result[COL_OPT_SUCCESS] = True
        execution_result[COL_OPT_FAILURE_REASON] = None
    except validation.UnsupportedQueryError as e:
        execution_result = _failed_execution_result(query, optimization_pipeline.target_database(), repetitions)
        execution_result[COL_T_OPT] = np.nan
        execution_result[COL_OPT_SUCCESS] = False
        execution_result[COL_OPT_FAILURE_REASON] = e.features

    execution_result[COL_ORIG_QUERY] = query
    execution_result[COL_OPT_SETTINGS] = [optimization_pipeline.describe()] * repetitions
    return execution_result


def optimize_and_execute_workload(queries: Iterable[qal.SqlQuery] | workloads.Workload,
                                  optimization_pipeline: pb.OptimizationPipeline, *,
                                  workload_repetitions: int = 1,
                                  per_query_repetitions: int = 1,
                                  shuffled: bool = False,
                                  query_preparation: Optional[QueryPreparationService] = None,
                                  include_labels: bool = False,
                                  post_process: Optional[Callable[[ExecutionResult], None]] = None) -> pd.DataFrame:
    """This function combines the functionality of `execute_workload` and `optimize_query` in one utility.

    Refer to the documentation of both source functions for more information.
    """
    queries = _wrap_workload(queries)
    results = []
    current_execution_index = 1
    for i in range(workload_repetitions):
        current_repetition_results = []
        if shuffled:
            queries = queries.shuffle()

        for label, query in queries.entries():
            execution_result = optimize_and_execute_query(query, optimization_pipeline,
                                                          repetitions=per_query_repetitions,
                                                          query_preparation=query_preparation,
                                                          post_process=post_process)
            execution_result[COL_EXEC_IDX] = list(range(current_execution_index,
                                                        current_execution_index + per_query_repetitions))
            if include_labels:
                execution_result[COL_LABEL] = label

            current_repetition_results.append(execution_result)
            current_execution_index += per_query_repetitions

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

    result_df = pd.concat(results)
    target_labels = [COL_LABEL] if include_labels else []
    target_labels += [COL_EXEC_IDX, COL_QUERY,
                      COL_WORKLOAD_ITER, COL_REP,
                      COL_T_OPT, COL_T_EXEC, COL_RESULT,
                      COL_OPT_SUCCESS, COL_OPT_FAILURE_REASON,
                      COL_ORIG_QUERY, COL_OPT_SETTINGS, COL_DB_CONFIG]
    return result_df[target_labels]
