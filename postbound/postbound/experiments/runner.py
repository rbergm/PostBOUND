from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from postbound import postbound as pb
from postbound.db import db
from postbound.qal import qal


@dataclass
class ExecutionResult:
    result_set: Any
    execution_time: float = np.nan
    optimization_time: float = np.nan


class QueryPreparationService:
    def prepare_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        pass


def execute_query(query: qal.SqlQuery, database: db.Database, *,
                  repetitions: int = 1, query_preparation: QueryPreparationService | None = None) -> ExecutionResult:
    pass


def execute_workload(queries: Iterable[qal.SqlQuery], database: db.Database, *,
                     workload_repetitions: int = 1, per_query_repetitions: int = 1,
                     query_preparation: QueryPreparationService | None = None) -> list[ExecutionResult]:
    pass


def optimize_and_execute_query(query: qal.SqlQuery, optimization_pipeline: pb.OptimizationPipeline, *,
                               repetitions: int = 1,
                               query_preparation: QueryPreparationService | None = None) -> ExecutionResult:
    pass


def optimize_and_execute_workload(queries: Iterable[qal.SqlQuery], optimization_pipeline: pb.OptimizationPipeline,
                                  workload_repetitions: int = 1, per_query_repetitions: int = 1,
                                  query_preparation: QueryPreparationService | None = None) -> ExecutionResult:
    pass
