"""Utilities to optimize and execute queries and workloads in a reproducible and transparent manner."""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import natsort
import numpy as np
import pandas as pd

from .. import _stages, util
from .._pipelines import OptimizationPipeline
from ..db._db import (
    Database,
    PrewarmingSupport,
    StopwatchSupport,
    TimeoutSupport,
    simplify_result_set,
)
from ..qal import transform
from ..qal._qal import Explain, SqlQuery
from .workloads import Workload, generate_workload

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


PredefLogger = Literal["tqdm"]


@dataclass
class ExecutionResult:
    """Captures all relevant components of a query optimization and execution result."""

    query: SqlQuery
    """The query that was executed. If the query was optimized and transformed, these modifications are included."""

    result_set: object = None
    """The raw result set of the query, if it was executed"""

    optimization_time: float = np.nan
    """The time in seconds it took to optimized the query by PostBOUND.

    This does not account for optimization by the actual database system and depends heavily on the quality of the
    implementation of the optimization strategies.
    """

    execution_time: float = np.nan
    """The time in seconds it took to execute the (potentially optimized) query by the actual database system.

    This execution time includes the entire end-to-end processing, i.e. starting with supplying the query to the
    database until the last byte of the result set was transferred back to PostBOUND. Therefore, this duration also
    includes the optimization time by the database system, as well as the entire time for data transfer.

    A value of *Inf* indicates that the query did not complete successfully and was cancelled due to a timeout. *NaN* encodes
    a failure in the execution process.
    """


class QueryPreparationService:
    """This service handles transformations of input queries that are executed before running the query.

    These transformations mostly ensure that all queries in a workload provide the same type of result even in face
    of input queries that are structured slightly differently. For example, the preparation service can transform
    all the queries to be executed as *EXPLAIN* or *COUNT(\\*)* queries. Furthermore, the preparation service can
    store SQL statements that have to be executed before running the query. For example, a statement that disables
    parallel execution could be supplied here.

    Parameters
    ----------
    explain : bool, optional
        Whether to force all queries to be executed as *EXPLAIN* queries, by default *False*
    count_star : bool, optional
        Whether to force all queries to be executed as *COUNT(\\*)* queries, overwriting their default projection. Defaults to
        *False*
    analyze : bool, optional
        Whether to force all queries to be executed as ``EXPLAIN ANALYZE`` queries. Setting this option implies `explain`,
        which therefore does not need to set manually. Defaults to *False*
    prewarm : bool, optional
        For database systems that support prewarming, this inflates the buffer pool with pages from the prepared query.
    preparatory_statements : Optional[list[str]], optional
        Statements that are executed as-is on the database connection before running the query, by default *None*

    See Also
    --------
    db.PrewarmingSupport : Technical details on how prewarming is implemented in PostBOUND
    """

    def __init__(
        self,
        *,
        explain: bool = False,
        count_star: bool = False,
        analyze: bool = False,
        prewarm: bool = False,
        preparatory_statements: Optional[list[str]] = None,
    ):
        self.explain = explain
        self.analyze = analyze
        self.count_star = count_star
        self.preparatory_stmts = (
            preparatory_statements if preparatory_statements else []
        )

        if explain and not analyze:
            if prewarm:
                warnings.warn(
                    "Ignoring prewarm setting since queries are only explained. Set prewarm manually to overwrite."
                )
            self.prewarm = False
        else:
            self.prewarm = prewarm

    def prepare_query(self, query: SqlQuery, *, on: Database) -> SqlQuery:
        """Applies the selected transformations to the given input query and executes the preparatory statements

        Parameters
        ----------
        query : SqlQuery
            The query to prepare
        on : Database
            The database to execute the preparatory statements on

        Returns
        -------
        SqlQuery
            The prepared query
        """
        if self.analyze:
            query = transform.as_explain(query, Explain.explain_analyze())
        elif self.explain:
            query = transform.as_explain(query, Explain.plan())

        if self.count_star:
            query = transform.as_count_star_query(query)

        if self.prewarm:
            if not isinstance(on, PrewarmingSupport):
                warnings.warn(
                    "Ignoring prewarm setting since the database does not support prewarming"
                )
            else:
                on.prewarm_tables(query.tables())

        for stmt in self.preparatory_stmts:
            on.execute_query(stmt, cache_enabled=False)

        return query


def _failed_execution_result(
    query: SqlQuery, database: Database, repetitions: int = 1
) -> pd.DataFrame:
    """Constructs a dummy data frame / row for queries that failed the execution.

    This data frame can be included in the overall result data frame as a replacement of the original data frame that would
    have been inserted if the query were executed successfully. It contains exactly the same number of rows and columns as the
    "correct" data, just with values that indicate failure.

    Parameters
    ----------
    query : SqlQuery
        The query that failed the execution
    database : Database
        The database on which the execution failed
    repetitions : int, optional
        The number of repetitions that should have been used for the query, by default 1

    Returns
    -------
    pd.DataFrame
        The data frame for the failed query
    """
    return pd.DataFrame(
        {
            COL_QUERY: [transform.drop_hints(query)] * repetitions,
            COL_QUERY_HINTS: [query.hints] * repetitions,
            COL_T_EXEC: [np.nan] * repetitions,
            COL_RESULT: [np.nan] * repetitions,
            COL_REP: list(range(1, repetitions + 1)),
            COL_DB_CONFIG: [database.describe()] * repetitions,
        }
    )


def _invoke_post_process(
    execution_result: ExecutionResult,
    action: Optional[Callable[[ExecutionResult], None]] = None,
) -> None:
    """Handler to run arbitrary post-process actions after a query was executed.

    Parameters
    ----------
    execution_result : ExecutionResult
        The result of the query execution
    action : Optional[Callable[[ExecutionResult], None]], optional
        The post-process handler. It receives the execution result as input and does not produce any output. If this is
        *None*, no post-processing is executed
    """
    if not action:
        return
    action(execution_result)


def execute_query(
    query: SqlQuery,
    database: Database,
    *,
    repetitions: int = 1,
    query_preparation: Optional[QueryPreparationService] = None,
    post_process: Optional[Callable[[ExecutionResult], None]] = None,
    timeout: Optional[float] = None,
    label: str = "",
    logger: Optional[PredefLogger] = None,
    optimization_time: float = math.nan,
) -> pd.DataFrame:
    """Runs the given query on the provided database.

    The query execution will be repeated for a total of `repetitions` times. Before the first repetition, the
    preparation service will be invoked if it has been supplied.

    In addition to the query execution, this function also accepts a `post_process` parameter. This parameter
    is a callable, that will be executed after each query run to perform arbitrary actions (e.g. online training of
    learned models). The callable receives an `ExecutionResult` object as input, but its output will not be processed in any
    way.

    The resulting data frame will be structured as follows:

    - the data frame will contain one row per query repetition
    - the current repetition is indicated in the `COL_REP` column
    - the actual query (without hints) is provided in the `COL_QUERY` column
    - the query hints are contained in the `COL_QUERY_HINTS` column
    - the execution time (end-to-end, i.e. until the last byte of the result set has been transferred back to
      PostBOUND) is provided in the `COL_T_EXEC` column (in seconds)
    - the query result of each repetition is contained in the `COL_RESULT` column

    Parameters
    ----------
    query : SqlQuery
        The query to execute
    database : Database
        The target database on which to execute the query
    repetitions : int, optional
        The number of times the query should be executed, by default 1
    query_preparation : Optional[QueryPreparationService], optional
        Preparation steps that should be performed before running the query. The preparation result will be used in place of
        the original query for all repetitions. Defaults to *None*, which means "no preparation".
    post_process : Optional[Callable[[ExecutionResult], None]], optional
        A post-process action that should be executed after each repetition of the query has been completed. Defaults to
        *None*, which means no post-processing.
    timeout : Optional[float], optional
        The maximum time in seconds that the query is allowed to run. If the query exceeds this time, the execution is
        cancelled and the execution time is set to *Inf*. If this parameter is omitted, no timeout is enforced. Notice that
        timeouts require the database to implement `TimeoutSupport`.
    label : str, optional
        A label that identifies the query. Currently, this is only used for logging purposes.
    logger : Optional[PredefLogger], optional
        Configures how progress of the repetitions should be logged.
    optimization_time : float, optional
        The optimization time that has been spent to generate the input query. If specified, a corresponding column is added
        to the output data frame.

    Returns
    -------
    pd.DataFrame
        The execution results for the input query

    See Also
    --------
    optimize_and_execute_query
    """
    original_query = query

    if logger == "tqdm":
        from tqdm import tqdm

        label = label if label else "query"
        iterations = tqdm(range(repetitions), desc=label, unit="rep", leave=False)
    else:
        iterations = range(repetitions)

    if query_preparation:
        query = query_preparation.prepare_query(query, on=database)

    if timeout is not None and not isinstance(database, TimeoutSupport):
        raise ValueError(f"Database system {database} does not provide timeout support")

    query_results = []
    execution_times = []

    for __ in iterations:
        if timeout is not None:
            start_time = time.perf_counter_ns()
            current_result = database.execute_with_timeout(query, timeout=timeout)
            end_time = time.perf_counter_ns()
            current_result = simplify_result_set(current_result)
            exec_time = (
                math.inf if current_result is None else (end_time - start_time) / 10**9
            )  # convert to seconds
        else:
            start_time = time.perf_counter_ns()
            current_result = database.execute_query(query, cache_enabled=False)
            end_time = time.perf_counter_ns()
            exec_time = (end_time - start_time) / 10**9  # convert to seconds

        if isinstance(database, StopwatchSupport):
            exec_time = database.last_query_runtime()

        query_results.append(current_result)
        execution_times.append(exec_time)
        execution_result = ExecutionResult(
            query, current_result, optimization_time, exec_time
        )
        _invoke_post_process(execution_result, post_process)

    return pd.DataFrame(
        {
            COL_QUERY: [transform.drop_hints(original_query)] * repetitions,
            COL_QUERY_HINTS: [original_query.hints] * repetitions,
            COL_T_EXEC: execution_times,
            COL_RESULT: query_results,
            COL_REP: list(range(1, repetitions + 1)),
            COL_DB_CONFIG: [database.describe()] * repetitions,
        }
    )


def _wrap_workload(
    queries: Iterable[SqlQuery] | Workload,
) -> Workload:
    """Transforms an iterable of queries into a proper workload object to enable execution by the runner methods.

    Parameters
    ----------
    queries : Iterable[SqlQuery] | Workload
        The queries to run as a workload. Can already be a workload object, which makes any transformation unnecessary.

    Returns
    -------
    Workload
        A workload of the given queries.
    """
    return queries if isinstance(queries, Workload) else generate_workload(queries)


def execute_workload(
    queries: Iterable[SqlQuery] | Workload,
    database: Database,
    *,
    workload_repetitions: int = 1,
    per_query_repetitions: int = 1,
    shuffled: bool = False,
    query_preparation: Optional[QueryPreparationService] = None,
    timeout: Optional[float] = None,
    include_labels: bool = False,
    post_process: Optional[Callable[[ExecutionResult], None]] = None,
    post_repetition_callback: Optional[Callable[[int], None]] = None,
    progressive_output: Optional[str | Path] = None,
    logger: Optional[Callable[[str], None] | PredefLogger] = None,
) -> pd.DataFrame:
    """Executes all the given queries on the provided database.

    Most of this process delegates to the `execute_query` method, so refer to its documentation for more details.
    The following are additional parameters for the workload execution:

    The entire workload will be repeated for an entire of `workload_repetitions` times. If `shuffled` is `True`,
    the order in which the queries are executed will be randomized _before_ each workload iteration. Note that this
    also applies for just a single repetition. The `include_labels` parameter expands the resulting data frame with
    another column that contains the query labels as obtained from the workload (integers are used if queries are
    supplied as an iterable rather than a workload object).

    In addition to the columns produced by `execute_query`, the resulting data frame will have the following extra columns:

    - an absolute index indicating which query is being executed (the first query has index 1, the query executed
      after that has index 2 and so on). This index will be increased for each query and workload repetition. I.e.,
      if a workload contains two queries, each query should be repeated 3 times and the entire workload should be
      repeated 4 times, the indexes will be 1..24. This information is contained in the `COL_EXEC_INDEX` column
    - the `COL_LABEL` column contains the query label if requested
    - the current workload iteration is described in the `COL_WORKLOAD_ITER` column

    Parameters
    ----------
    queries : Iterable[SqlQuery] | Workload
        The workload to execute
    database : Database
        The target database on which to execute the query
    workload_repetitions : int, optional
        The number of times the entire workload should be repeated, by default ``1``
    per_query_repetitions : int, optional
        The number of times each query should be repeated within each workload repetition. The per-query repetitions happen
        sequentially one after another before transitioning to the next query. Defaults to ``1``.
    shuffled : bool, optional
        Whether to randomize the execution order of each query within the workload. Shuffling is applied before each workload
        repetition. Per query repetitions are *not* influenced by this setting.
    query_preparation : Optional[QueryPreparationService], optional
        Preparation steps that should be performed before running the query. The preparation result will be used in place of
        the original query for all repetitions. Defaults to *None*, which means "no preparation".
    timeout : Optional[float], optional
        The maximum time in seconds that the query is allowed to run. If the query exceeds this time, the execution is
        cancelled and the execution time is set to *Inf*. If this parameter is omitted, no timeout is enforced. Notice that
        timeouts require the database to implement `TimeoutSupport`.
    include_labels : bool, optional
        Whether to add the label of each query to the workload results, by default *False*
    post_process : Optional[Callable[[ExecutionResult], None]], optional
        A post-process action that should be executed after each repetition of the query has been completed. Defaults to
        *None*, which means no post-processing.
    post_repetition_callback : Optional[Callable[[int], None]], optional
        An optional post-process action that is executed after each workload repetition. The current repetition number is
        provided as the only argument. Repetitions start at 0.
    progressive_output: Optional[str | Path], optional
        If provided, the results will be written to this file after each workload repetition. If the file already exists, it
        will be overwritten. This is file is assumed to be a CSV file.
    logger : Optional[Callable[[str], None] | PredefLogger], optional
        Configures how progress should be logged. Depending on the specific argument, a number of different strategies are
        available:

        - passing *None* (the default) disables logging
        - passing a callable invokes the function before every query execution. It receives information about the current
          execution as argument
        - referencing a pre-defined logger invokes. Currently, only *tqdm*  is supported. It uses the same library to
          print a progress bar

    Returns
    -------
    pd.DataFrame
        The execution results for the input workload

    See Also
    --------
    execute_query
    """
    queries = _wrap_workload(queries)
    progressive_output = Path(progressive_output) if progressive_output else None
    use_tqdm = logger == "tqdm"
    logger = util.make_logger(False) if use_tqdm or logger is None else logger
    if use_tqdm:
        from tqdm import tqdm

    results: list[pd.DataFrame] = []
    current_execution_index = 1
    for i in range(workload_repetitions):
        current_repetition_results: list[pd.DataFrame] = []
        if shuffled:
            queries = queries.shuffle()

        workload_entries = (
            tqdm(
                queries.entries(),
                desc=f"Repetition {i + 1}/{workload_repetitions}",
                unit="q",
            )
            if use_tqdm
            else queries.entries()
        )

        for label, query in workload_entries:
            logger(
                f"Now benchmarking query {label} (repetition {i + 1}/{workload_repetitions})"
            )
            execution_result = execute_query(
                query,
                database,
                repetitions=per_query_repetitions,
                timeout=timeout,
                query_preparation=query_preparation,
                post_process=post_process,
                label=label,
                logger="tqdm" if use_tqdm else None,
            )
            execution_result[COL_EXEC_IDX] = list(
                range(
                    current_execution_index,
                    current_execution_index + per_query_repetitions,
                )
            )
            if include_labels:
                execution_result[COL_LABEL] = label

            current_repetition_results.append(execution_result)
            current_execution_index += per_query_repetitions

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

        if progressive_output and progressive_output.is_file():
            current_output = prepare_export(current_df)
            current_output.to_csv(
                progressive_output, mode="a", index=False, header=False
            )
        elif progressive_output:
            current_output = prepare_export(current_df)
            current_output.to_csv(progressive_output, index=False, mode="w")

        if post_repetition_callback:
            post_repetition_callback(i)

    result_df = pd.concat(results)
    target_labels = [COL_LABEL] if include_labels else []
    target_labels += [
        COL_EXEC_IDX,
        COL_QUERY,
        COL_WORKLOAD_ITER,
        COL_REP,
        COL_T_EXEC,
        COL_RESULT,
    ]
    return result_df[target_labels].reset_index(drop=True)


def optimize_and_execute_query(
    query: SqlQuery,
    optimization_pipeline: OptimizationPipeline,
    *,
    repetitions: int = 1,
    query_preparation: Optional[QueryPreparationService] = None,
    post_process: Optional[Callable[[ExecutionResult], None]] = None,
    timeout: Optional[float] = None,
    label: str = "",
    logger: Optional[PredefLogger] = None,
) -> pd.DataFrame:
    """Optimizes the a query according to the settings of an optimization pipeline and executes it afterwards.

    This function delegates most of its work to `execute_query`. In addition, the resulting data frame contains the
    following columns:

    - the time it took PostBOUND to optimize the query is indicated in the `COL_T_OPT` column (in seconds)
    - whether the query was optimized successfully is indicated in the `COL_OPT_SUCCESS` column
    - if the optimization failed, the reason(s) for the failure are contained in the `COL_OPT_FAILURE_REASON` column
    - the original query (i.e. before optimization and query preparation) is provided in the `COL_ORIG_QUERY` column
    - the selected optimization settings of the optimization pipeline is contained in the `COL_OPT_SETTINGS` column

    Parameters
    ----------
    query : SqlQuery
        The query to execute
    optimization_pipeline : OptimizationPipeline
        The optimization settings that should be used to optimize the given query. The pipeline is also used to extract the
        target database which is used to execute the query
    repetitions : int, optional
        The number of times the query should be executed, by default ``1``. The query is optimized only once right before the
        first execution.
    query_preparation : Optional[QueryPreparationService], optional
        Preparation steps that should be performed before running the query. The preparation result will be used in place of
        the original query for all repetitions. Defaults to *None*, which means "no preparation".
    post_process : Optional[Callable[[ExecutionResult], None]], optional
        A post-process action that should be executed after each repetition of the query has been completed. Defaults to
        *None*, which means no post-processing.
    timeout : Optional[float], optional
        The maximum time in seconds that the query is allowed to run. If the query exceeds this time, the execution is
        cancelled and the execution time is set to *Inf*. If this parameter is omitted, no timeout is enforced. Notice that
        timeouts require the database to implement `TimeoutSupport`.
    label : str, optional
        A label that identifies the query. Currently, this is only used for logging purposes.
    logger : Optional[PredefLogger], optional
        Configures how progress of the repetitions should be logged.

    Returns
    -------
    pd.DataFrame
        The optimization and execution results for the given query

    See Also
    --------
    execute_query
    """
    try:
        start_time = time.perf_counter_ns()
        optimized_query = optimization_pipeline.optimize_query(query)
        end_time = time.perf_counter_ns()
        optimization_time = (
            end_time - start_time
        ) / 1_000_000_000  # convert to seconds

        execution_result = execute_query(
            optimized_query,
            repetitions=repetitions,
            query_preparation=query_preparation,
            database=optimization_pipeline.target_database(),
            post_process=post_process,
            timeout=timeout,
            optimization_time=optimization_time,
            label=label,
            logger=logger,
        )
        execution_result[COL_T_OPT] = optimization_time
        execution_result[COL_OPT_SUCCESS] = True
        execution_result[COL_OPT_FAILURE_REASON] = None
    except _stages.UnsupportedQueryError as e:
        execution_result = _failed_execution_result(
            query, optimization_pipeline.target_database(), repetitions
        )
        execution_result[COL_T_OPT] = np.nan
        execution_result[COL_OPT_SUCCESS] = False
        execution_result[COL_OPT_FAILURE_REASON] = e.features

    execution_result[COL_ORIG_QUERY] = query
    pipeline_desc = optimization_pipeline.describe()
    execution_result[COL_OPT_SETTINGS] = [pipeline_desc] * repetitions
    return execution_result


def optimize_and_execute_workload(
    queries: Iterable[SqlQuery] | Workload,
    optimization_pipeline: OptimizationPipeline,
    *,
    workload_repetitions: int = 1,
    per_query_repetitions: int = 1,
    shuffled: bool = False,
    query_preparation: Optional[QueryPreparationService] = None,
    timeout: Optional[float] = None,
    include_labels: bool = False,
    post_process: Optional[Callable[[ExecutionResult], None]] = None,
    post_repetition_callback: Optional[Callable[[int], None]] = None,
    progressive_output: Optional[str | Path] = None,
    logger: Optional[Callable[[str], None] | PredefLogger] = None,
) -> pd.DataFrame:
    """This function combines the functionality of `execute_workload` and `optimize_query` in one utility.

    Each workload iteration starts "from scratch", i.e. with the raw, un-optimized queries. If the post-process actions mutated
    some state, these mutations will however still be reflected.

    Refer to the documentation of the methods under *See Also* for documentation of the provided data frame and execution/
    optimization details.

    Parameters
    ----------
    queries : Iterable[SqlQuery] | Workload
        The queries that should be optimized and benchmarked
    optimization_pipeline : OptimizationPipeline
        The optimization settings that should be used to optimize the given workload. The pipeline is also used to extract the
        target database which is used to execute the queries
    workload_repetitions : int, optional
        The number of times the entire workload should be repeated, by default ``1``. During each repetition, the workload
        will be optimized anew.
    per_query_repetitions : int, optional
        The number of times each query should be repeated within each workload repetition. The per-query repetitions happen
        sequentially one after another before transitioning to the next query. Defaults to ``1``.
    shuffled : bool, optional
        Whether to randomize the execution order of each query within the workload. Shuffling is applied before each workload
        repetition. Per query repetitions are *not* influenced by this setting.
    query_preparation : Optional[QueryPreparationService], optional
        Preparation steps that should be performed before running the query. The preparation result will be used in place of
        the original query for all repetitions. Defaults to *None*, which means "no preparation".
    timeout : Optional[float], optional
        The maximum time in seconds that the query is allowed to run. If the query exceeds this time, the execution is
        cancelled and the execution time is set to *Inf*. If this parameter is omitted, no timeout is enforced. Notice that
        timeouts require the database to implement `TimeoutSupport`.
    include_labels : bool, optional
        Whether to add the label of each query to the workload results, by default *False*
    post_process : Optional[Callable[[ExecutionResult], None]], optional
        A post-process action that should be executed after each repetition of the query has been completed. Defaults to
        *None*, which means no post-processing.
    post_repetition_callback : Optional[Callable[[int], None]], optional
        An optional post-process action that is executed after each workload repetition. The current repetition number is
        provided as the only argument. Repetitions start at 0.
    progressive_output: Optional[str | Path], optional
        If provided, the results will be written to this file after each workload repetition. If the file already exists, it
        will be overwritten. This is file is assumed to be a CSV file.
    logger : Optional[Callable[[str], None] | PredefLogger], optional
        Configures how progress should be logged. Depending on the specific argument, a number of different strategies are
        available:

        - passing *None* (the default) disables logging
        - passing a callable invokes the function before every query execution. It receives information about the current
          execution as argument
        - referencing a pre-defined logger invokes. Currently, only *tqdm*  is supported. It uses the same library to
          print a progress bar

    Returns
    -------
    pd.DataFrame
        The optimization and execution results for the given workload

    See Also
    --------
    execute_and_optimize_query
    execute_workload
    """
    queries = _wrap_workload(queries)
    progressive_output = Path(progressive_output) if progressive_output else None
    use_tqdm = logger == "tqdm"
    logger = util.make_logger(False) if use_tqdm or logger is None else logger
    if use_tqdm:
        from tqdm import tqdm

    results: list[pd.DataFrame] = []
    current_execution_index = 1
    for i in range(workload_repetitions):
        current_repetition_results: list[pd.DataFrame] = []
        if shuffled:
            queries = queries.shuffle()

        workload_entries = (
            tqdm(
                queries.entries(),
                desc=f"Repetition {i + 1}/{workload_repetitions}",
                unit="q",
            )
            if use_tqdm
            else queries.entries()
        )

        for label, query in workload_entries:
            logger(
                f"Now benchmarking query {label} (repetition {i + 1}/{workload_repetitions})"
            )
            execution_result = optimize_and_execute_query(
                query,
                optimization_pipeline,
                repetitions=per_query_repetitions,
                timeout=timeout,
                query_preparation=query_preparation,
                post_process=post_process,
                label=label,
                logger="tqdm" if use_tqdm else None,
            )
            execution_result[COL_EXEC_IDX] = list(
                range(
                    current_execution_index,
                    current_execution_index + per_query_repetitions,
                )
            )
            if include_labels:
                execution_result[COL_LABEL] = label

            current_repetition_results.append(execution_result)
            current_execution_index += per_query_repetitions

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

        if progressive_output and progressive_output.is_file():
            current_output = prepare_export(current_df)
            current_output.to_csv(
                progressive_output, mode="a", index=False, header=False
            )
        elif progressive_output:
            current_output = prepare_export(current_df)
            current_output.to_csv(progressive_output, index=False, mode="w")

        if post_repetition_callback:
            post_repetition_callback(i)

    result_df = pd.concat(results)
    target_labels = [COL_LABEL] if include_labels else []
    target_labels += [
        COL_EXEC_IDX,
        COL_QUERY,
        COL_WORKLOAD_ITER,
        COL_REP,
        COL_T_OPT,
        COL_T_EXEC,
        COL_RESULT,
        COL_OPT_SUCCESS,
        COL_OPT_FAILURE_REASON,
        COL_ORIG_QUERY,
        COL_OPT_SETTINGS,
        COL_DB_CONFIG,
    ]
    return result_df[target_labels].reset_index(drop=True)


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

    example_result = prepared_df[COL_RESULT].iloc[0]
    if (
        isinstance(example_result, list)
        or isinstance(example_result, tuple)
        or isinstance(example_result, dict)
    ):
        prepared_df[COL_RESULT] = prepared_df[COL_RESULT].apply(json.dumps)

    if COL_OPT_SETTINGS in prepared_df:
        prepared_df[COL_OPT_SETTINGS] = prepared_df[COL_OPT_SETTINGS].apply(
            util.to_json
        )
    if COL_DB_CONFIG in prepared_df:
        prepared_df[COL_DB_CONFIG] = prepared_df[COL_DB_CONFIG].apply(util.to_json)

    return prepared_df


def sort_results(
    results_df: pd.DataFrame, by_column: str | tuple[str] = (COL_LABEL, COL_EXEC_IDX)
) -> pd.DataFrame:
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
        The columns by which to order, by default `(COL_LABEL, COL_EXEC_IDX)`. A lexicographic ordering will
        be applied to all of them.

    Returns
    -------
    pd.DataFrame
        A reordered data frame. The original data frame is not modified
    """
    return results_df.sort_values(
        by=by_column, key=lambda series: np.argsort(natsort.index_natsorted(series))
    )
