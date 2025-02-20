from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Optional

from ..validation import OptimizationPreCheck
from ... import db, util
from ..._core import (TableReference, ScanOperator, JoinOperator)
from ..._qep import QueryPlan, SortKey
from ..._stages import (PlanEnumerator, CostModel, CardinalityEstimator)
from ...qal import SqlQuery
from ...db.postgres import (PostgresScanHints, PostgresJoinHints)
from ...util import jsondict


class DynamicProgrammingEnumerator(PlanEnumerator):

    def __init__(self, *, supported_scan_ops: Optional[set[ScanOperator]] = None,
                 supported_join_ops: Optional[set[JoinOperator]] = None,
                 target_db: Optional[db.Database] = None) -> None:
        target_db = target_db if target_db is not None else db.DatabasePool.get_instance().current_database()

        supported_scan_ops = supported_scan_ops if supported_scan_ops is not None else set(ScanOperator)
        supported_join_ops = supported_join_ops if supported_join_ops is not None else set(JoinOperator)

        if target_db is not None:
            supported_scan_ops = {op for op in supported_scan_ops if target_db.hinting().supports_hint(op)}
            supported_join_ops = {op for op in supported_join_ops if target_db.hinting().supports_hint(op)}

        self._target_db = target_db
        self._scan_ops = supported_scan_ops
        self._join_ops = supported_join_ops

    def generate_execution_plan(self, query, *, cost_model, cardinality_estimator) -> QueryPlan:
        cost_model.initialize(self._target_db, query)
        cardinality_estimator.initialize(self._target_db, query)

        dp_table: dict[frozenset[TableReference], QueryPlan] = {}
        self._determine_base_access_paths(dp_table, query, cost_model=cost_model, cardinality_estimator=cardinality_estimator)
        final_plan = self._build_join_paths(dp_table, query, cost_model=cost_model,
                                            cardinality_estimator=cardinality_estimator)

        cost_model.cleanup()
        cardinality_estimator.cleanup()
        return final_plan

    def pre_check(self) -> OptimizationPreCheck:
        # TODO: restrict to simple inner equi-joins for now
        return super().pre_check()

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
            candidate_plans: list[QueryPlan] = []

            if ScanOperator.SequentialScan in self._scan_ops:
                candidate_plans.append(QueryPlan(ScanOperator.SequentialScan, table=table))
            candidate_plans += self._determine_index_paths(query, table)

            candidate_plans = [candidate.with_estimates(cardinality=cardinality_estimator.calculate_estimate(query, table))
                               for candidate in candidate_plans]
            candidate_plans = [candidate.with_estimates(cost=cost_model.estimate_cost(query, candidate))
                               for candidate in candidate_plans]

            cheapest_plan = min(candidate_plans, key=lambda plan: plan.estimated_cost)
            dp_table[frozenset([table])] = cheapest_plan

    def _determine_index_paths(self, query: SqlQuery, table: TableReference) -> Iterable[QueryPlan]:
        required_columns = query.columns_of(table)
        can_idx_only_scan = len(required_columns) <= 1  # check for <= 1 to include cross products with select star
        candidate_indexes = {column: self._target_db.schema().indexes_on(column) for column in required_columns}

        if not candidate_indexes:
            return []

        candidate_plans: list[QueryPlan] = []
        for column, available_indexes in candidate_indexes.items():
            if not available_indexes:
                continue
            sorting = [SortKey.of(column)]

            for index in available_indexes:
                if ScanOperator.IndexScan in self._scan_ops:
                    candidate_plans.append(QueryPlan(ScanOperator.IndexScan, table=table, index=index, sort_keys=sorting))
                if can_idx_only_scan and ScanOperator.IndexOnlyScan in self._scan_ops:
                    candidate_plans.append(QueryPlan(ScanOperator.IndexOnlyScan, table=table, index=index, sort_keys=sorting))

        if ScanOperator.BitmapScan in self._scan_ops:
            # the target DB is responsible for figuring out good bitmap index hierarchies
            candidate_plans.append(QueryPlan(ScanOperator.BitmapScan, table=table, indexes=candidate_indexes))

        return candidate_plans

    def _build_join_paths(self, dp_table: dict[frozenset[TableReference], QueryPlan], query: SqlQuery, *,
                          cost_model: CostModel, cardinality_estimator: CardinalityEstimator) -> QueryPlan:
        predicates = query.predicates()
        candidate_tables = query.tables()

        for current_level in range(2, len(candidate_tables) + 1):
            valid_intermediates = itertools.combinations(candidate_tables, current_level)
            access_paths = {frozenset(join): self._access_path_for_intermediate(dp_table, join,
                                                                                query=query,
                                                                                cost_model=cost_model,
                                                                                cardinality_estimator=cardinality_estimator)
                            for join in valid_intermediates if predicates.joins_tables(join)}
            dp_table.update(access_paths)

        return dp_table[frozenset(candidate_tables)]

    def _access_path_for_intermediate(self, dp_table: dict[frozenset[TableReference], QueryPlan],
                                      intermediate: Iterable[TableReference], *,
                                      query: SqlQuery,
                                      cost_model: CostModel, cardinality_estimator: CardinalityEstimator) -> QueryPlan:
        intermediate = set(intermediate)
        candidate_plans: list[QueryPlan] = []

        for outer in util.collections.powerset(intermediate):
            if not outer or len(outer) == len(intermediate):
                continue

            inner = intermediate - outer
            outer_plan = dp_table[frozenset(outer)]
            inner_plan = dp_table[frozenset(inner)]

            if JoinOperator.NestedLoopJoin in self._join_ops:
                candidate_plans.append(QueryPlan(JoinOperator.NestedLoopJoin, children=[outer_plan, inner_plan]))

            if JoinOperator.HashJoin in self._join_ops:
                candidate_plans.append(QueryPlan(JoinOperator.HashJoin, children=[outer_plan, inner_plan]))

            if JoinOperator.SortMergeJoin in self._join_ops:
                # the target DB is utimately responsible for figuring out whether it needs explicit sorts or whether it can
                # just merge directly
                candidate_plans.append(QueryPlan(JoinOperator.SortMergeJoin, children=[outer_plan, inner_plan]))

        candidate_plans = [candidate.with_estimates(cardinality=cardinality_estimator.calculate_estimate(query, candidate))
                           for candidate in candidate_plans]
        candidate_plans = [candidate.with_estimates(cost=cost_model.estimate_cost(query, candidate))
                           for candidate in candidate_plans]

        return min(candidate_plans, key=lambda plan: plan.estimated_cost)


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
