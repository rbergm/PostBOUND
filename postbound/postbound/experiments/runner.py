from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np

from postbound import postbound as pb
from postbound.db import db
from postbound.qal import qal, transform, clauses


# TODO: add shuffled execution
# TODO: we should record intermediate timings even for per query repetitions
# TODO: maybe it is better to rely on dataframes altogether, this makes storing original queries, optimization settings, etc much easier


@dataclass
class ExecutionResult:
    result_set: Any
    execution_time: float = np.nan
    optimization_time: float = np.nan


class WorkloadResults(collections.UserDict[qal.SqlQuery, list[ExecutionResult]]):
    def __init__(self) -> None:
        super().__init__()
        self.queries = []

    def add_result(self, query: qal.SqlQuery, result: ExecutionResult) -> None:
        if query not in self.data:
            self.queries.append(query)
            self.data[query] = [result]
        else:
            self.data[query].append(result)

    def results(self) -> Iterable[tuple[qal.SqlQuery, list[ExecutionResult]]]:
        return [(query, self.data[query]) for query in self.queries]


class QueryPreparationService:
    def __init__(self, *, explain: bool, count_star: bool, analyze: bool = False):
        self.explain = explain
        self.analyze = analyze
        self.count_star = count_star

    def prepare_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        if self.analyze:
            query = transform.as_explain(query, clauses.Explain.explain_analyze())
        elif self.explain:
            query = transform.as_explain(query, clauses.Explain.pure())

        if self.count_star:
            query = transform.as_count_star_query(query)

        return query


def execute_query(query: qal.SqlQuery, database: db.Database, *,
                  repetitions: int = 1, query_preparation: QueryPreparationService | None = None) -> ExecutionResult:
    if query_preparation:
        query = query_preparation.prepare_query(query)

    best_exec_time = timedelta.max
    best_result_set = None
    for __ in range(repetitions):
        start_time = datetime.now()
        current_result = database.execute_query(query, cache_enabled=False)
        end_time = datetime.now()
        exec_time = end_time - start_time
        if exec_time < best_exec_time:
            best_result_set = current_result
            best_exec_time = exec_time

    return ExecutionResult(best_result_set, best_exec_time.total_seconds())


def execute_workload(queries: Iterable[qal.SqlQuery], database: db.Database, *,
                     workload_repetitions: int = 1, per_query_repetitions: int = 1,
                     query_preparation: QueryPreparationService | None = None) -> WorkloadResults:
    results = WorkloadResults()
    for __ in range(workload_repetitions):
        for query in queries:
            execution_result = execute_query(query, database, repetitions=per_query_repetitions,
                                             query_preparation=query_preparation)
            results.add_result(query, execution_result)
    return results


def optimize_and_execute_query(query: qal.SqlQuery, optimization_pipeline: pb.OptimizationPipeline, *,
                               repetitions: int = 1,
                               query_preparation: QueryPreparationService | None = None) -> ExecutionResult:
    start_time = datetime.now()
    optimized_query = optimization_pipeline.optimize_query(query)
    end_time = datetime.now()
    optimization_time = end_time - start_time

    execution_result = execute_query(optimized_query, repetitions=repetitions, query_preparation=query_preparation,
                                     database=optimization_pipeline.target_dbs.interface())
    execution_result.optimization_time = optimization_time
    return execution_result


def optimize_and_execute_workload(queries: Iterable[qal.SqlQuery], optimization_pipeline: pb.OptimizationPipeline,
                                  workload_repetitions: int = 1, per_query_repetitions: int = 1,
                                  query_preparation: QueryPreparationService | None = None
                                  ) -> WorkloadResults:
    results = WorkloadResults()
    for __ in range(workload_repetitions):
        for query in queries:
            query_result = optimize_and_execute_query(query, optimization_pipeline, repetitions=per_query_repetitions,
                                                      query_preparation=query_preparation)
            results.add_result(query, query_result)
    return results
