from __future__ import annotations

import math
from typing import Optional

from ... import db
from ..._core import (TableReference, ScanOperator, JoinOperator)
from ..._qep import QueryPlan
from ..._stages import (PlanEnumerator, CostModel, CardinalityEstimator)
from ...qal import SqlQuery
from ...db.postgres import (PostgresScanHints, PostgresJoinHints)
from ...util import jsondict


class DynamicProgrammingEnumerator(PlanEnumerator):

    def __init__(self, *, supported_scan_ops: Optional[set[ScanOperator]] = None,
                 supported_join_ops: Optional[set[JoinOperator]] = None,
                 target_db: Optional[db.Database] = None) -> None:
        raise NotImplementedError("The DynamicProgrammingEnumerator is not yet functional. "
                                  "Please use your own enumerator for now.")

        target_db = target_db if target_db is not None else db.DatabasePool.get_instance().current_database()

        supported_scan_ops = supported_scan_ops if supported_scan_ops is not None else set(ScanOperator)
        supported_join_ops = supported_join_ops if supported_join_ops is not None else set(JoinOperator)

        if target_db is not None:
            supported_scan_ops = {op for op in supported_scan_ops if target_db.hinting().supports_hint(op)}
            supported_join_ops = {op for op in supported_join_ops if target_db.hinting().supports_hint(op)}

        self._target_db = target_db
        self._scan_ops = supported_scan_ops
        self._join_ops = supported_join_ops

    def generate_execution_plan(self, query, *, cost_model, cardinality_estimator):
        cost_model.initialize(self._target_db, query)
        cardinality_estimator.initialize(self._target_db, query)

        dp_table: dict[frozenset[TableReference], QueryPlan] = {}
        self._determine_base_access_paths(dp_table, query, cost_model=cost_model, cardinality_estimator=cardinality_estimator)

        cost_model.cleanup()
        cardinality_estimator.cleanup()

    def describe(self) -> jsondict:
        return {
            "name": "dynamic_programming",
            "flavor": "default",
            "scan_ops": [op.name for op in self._scan_ops],
            "join_ops": [op.name for op in self._join_ops],
            "database_system": self._target_db.describe()
        }

    def _determine_base_access_paths(self, dp_table: dict[frozenset[TableReference], QueryPlan], query: SqlQuery, *,
                                     cost_model: CostModel, cardinality_estimator: CardinalityEstimator) -> None:
        for table in query.tables():
            cheapest_cost = math.inf
            cheapest_plan = None

            for scan_op in self._scan_ops:
                pass

            dp_table[frozenset([table])] = cheapest_plan


class PostgresDynProg(PlanEnumerator):
    """Dynamic programming-based plan enumeration strategy that mimics the behavior of the Postgres query optimizer.

    Postgres-style dynamic programming means two things: first, we use the Postgres pruning rules to reduce the search space.
    Second, we apply the same opinionated traversal rules. Most importantly, this concerns when we consider materialization or
    memoization of subplans. If some of the related operators are not allowed, the traversal rules are adjusted accordingly.

    The implementation is based on a translation of the actual Postgres source code.

    Parameters
    ----------
    supported_scan_ops : Optional[set[ScanOperators]], optional
        _description_, by default None
    supported_join_ops : Optional[set[JoinOperators]], optional
        _description_, by default None
    enable_materialize : bool, optional
        _description_, by default True
    enable_memoize : bool, optional
        _description_, by default True
    enable_sort : bool, optional
        _description_, by default True
    target_db : Optional[db.Database], optional
        _description_, by default None
    """

    def __init__(self, *, supported_scan_ops: Optional[set[ScanOperator]] = None,
                 supported_join_ops: Optional[set[JoinOperator]] = None,
                 enable_materialize: bool = True, enable_memoize: bool = True, enable_sort: bool = True,
                 target_db: Optional[db.Database] = None) -> None:
        raise NotImplementedError("The Postgres-style dynamic programming is not yet functional. "
                                  "Please use your own enumerator for now.")

        target_db = target_db if target_db is not None else db.DatabasePool.get_instance().current_database()

        supported_scan_ops = supported_scan_ops if supported_scan_ops is not None else PostgresScanHints
        supported_join_ops = supported_join_ops if supported_join_ops is not None else PostgresJoinHints

        if target_db is not None:
            supported_scan_ops = {op for op in supported_scan_ops if target_db.hinting().supports_hint(op)}
            supported_join_ops = {op for op in supported_join_ops if target_db.hinting().supports_hint(op)}

        self._target_db = target_db
        self._scan_ops = supported_scan_ops
        self._join_ops = supported_join_ops
        self._enable_materialize = enable_materialize
        self._enable_memoize = enable_memoize
        self._enable_sort = enable_sort

    def generate_execution_plan(self, query, *, cost_model, cardinality_estimator):
        raise NotImplementedError

    def describe(self):
        return {
            "name": "dynamic_programming",
            "flavor": "postgres",
            "scan_ops": [op.name for op in self._scan_ops],
            "join_ops": [op.name for op in self._join_ops],
            "database_system": self._target_db.describe()
        }
