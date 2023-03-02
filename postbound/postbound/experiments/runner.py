from __future__ import annotations

import random
from datetime import datetime
from typing import Iterable

import pandas as pd

from postbound import postbound as pb
from postbound.db import db
from postbound.qal import qal, transform, clauses

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


class QueryPreparationService:
    """

    Describes necessary modifications on the query
    and SQL statements that have to be executed before

    """

    def __init__(self, *, explain: bool = False, count_star: bool = False, analyze: bool = False,
                 preparatory_statements: list[str] | None = None):
        self.explain = explain
        self.analyze = analyze
        self.count_star = count_star
        self.preparatory_stmts = preparatory_statements if preparatory_statements else []

    def prepare_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        if self.analyze:
            query = transform.as_explain(query, clauses.Explain.explain_analyze())
        elif self.explain:
            query = transform.as_explain(query, clauses.Explain.pure())

        if self.count_star:
            query = transform.as_count_star_query(query)

        return query

    def preparatory_statements(self) -> list[str]:
        return self.preparatory_stmts


def execute_query(query: qal.SqlQuery, database: db.Database, *,
                  repetitions: int = 1, query_preparation: QueryPreparationService | None = None) -> pd.DataFrame:
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
        exec_time = end_time - start_time

        query_results.append(current_result)
        execution_times.append(exec_time.total_seconds())

    return pd.DataFrame({COL_QUERY: [transform.drop_hints(original_query)] * repetitions,
                         COL_QUERY_HINTS: [original_query.hints] * repetitions,
                         COL_T_EXEC: execution_times,
                         COL_RESULT: query_results,
                         COL_REP: range(1, repetitions + 1)})


def execute_workload(queries: Iterable[qal.SqlQuery], database: db.Database, *,
                     workload_repetitions: int = 1, per_query_repetitions: int = 1, shuffled: bool = False,
                     query_preparation: QueryPreparationService | None = None,
                     query_labels: dict[qal.SqlQuery, str] | None = None) -> pd.DataFrame:
    results = []
    current_execution_index = 0
    for i in range(workload_repetitions):
        current_repetition_results = []
        if shuffled:
            queries = random.sample(queries, k=len(queries))

        for query in queries:
            current_execution_index += 1
            execution_result = execute_query(query, database, repetitions=per_query_repetitions,
                                             query_preparation=query_preparation)
            execution_result[COL_EXEC_IDX] = list(range(current_execution_index,
                                                        current_execution_index + per_query_repetitions))
            current_repetition_results.append(execution_result)

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

    result_df = pd.concat(results)
    result_df[COL_LABEL] = result_df[COL_QUERY].apply(query_labels.get) if query_labels else ""
    return result_df[[COL_LABEL, COL_EXEC_IDX, COL_QUERY,
                      COL_WORKLOAD_ITER, COL_REP,
                      COL_T_EXEC, COL_RESULT]]


def optimize_and_execute_query(query: qal.SqlQuery, optimization_pipeline: pb.OptimizationPipeline, *,
                               repetitions: int = 1,
                               query_preparation: QueryPreparationService | None = None) -> pd.DataFrame:
    start_time = datetime.now()
    optimized_query = optimization_pipeline.optimize_query(query)
    end_time = datetime.now()
    optimization_time = end_time - start_time

    execution_result = execute_query(optimized_query, repetitions=repetitions, query_preparation=query_preparation,
                                     database=optimization_pipeline.target_dbs.interface())
    execution_result[COL_T_OPT] = optimization_time.total_seconds()
    execution_result[COL_ORIG_QUERY] = query
    execution_result[COL_OPT_SETTINGS] = optimization_pipeline

    return execution_result


def optimize_and_execute_workload(queries: Iterable[qal.SqlQuery], optimization_pipeline: pb.OptimizationPipeline, *,
                                  workload_repetitions: int = 1, per_query_repetitions: int = 1, shuffled: bool = False,
                                  query_preparation: QueryPreparationService | None = None,
                                  query_labels: dict[qal.SqlQuery, str] | None = None) -> pd.DataFrame:
    results = []
    current_execution_index = 0
    for i in range(workload_repetitions):
        current_repetition_results = []
        if shuffled:
            queries = random.sample(queries, k=len(queries))

        for query in queries:
            current_execution_index += 1
            execution_result = optimize_and_execute_query(query, optimization_pipeline,
                                                          repetitions=per_query_repetitions,
                                                          query_preparation=query_preparation)
            execution_result[COL_EXEC_IDX] = list(range(current_execution_index,
                                                        current_execution_index + per_query_repetitions))
            current_repetition_results.append(execution_result)

        current_df = pd.concat(current_repetition_results)
        current_df[COL_WORKLOAD_ITER] = i + 1
        results.append(current_df)

    result_df = pd.concat(results)
    result_df[COL_LABEL] = result_df[COL_ORIG_QUERY].apply(query_labels.get) if query_labels else ""
    return result_df[[COL_LABEL, COL_EXEC_IDX, COL_QUERY,
                      COL_WORKLOAD_ITER, COL_REP,
                      COL_T_OPT, COL_T_EXEC, COL_RESULT,
                      COL_ORIG_QUERY]]
