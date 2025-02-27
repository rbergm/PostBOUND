from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Optional

from .. import validation
from ..validation import OptimizationPreCheck
from ... import db, util
from ..._core import (TableReference, ScanOperator, JoinOperator)
from ..._qep import QueryPlan, SortKey
from ..._stages import (PlanEnumerator, CostModel, CardinalityEstimator)
from ...qal import SqlQuery
from ...db.postgres import (PostgresScanHints, PostgresJoinHints)
from ...util import jsondict


DPTable = dict[frozenset[TableReference], QueryPlan]


def _calc_plan_estimates(query: SqlQuery, plan: QueryPlan, *,
                         cost_model: CostModel, cardinality_estimator: CardinalityEstimator) -> QueryPlan:
    """Handler method to update the cost and cardinality estimates of a given plan."""
    card_est = cardinality_estimator.calculate_estimate(query, plan.tables())
    plan = plan.with_estimates(cardinality=card_est)
    cost_est = cost_model.estimate_cost(query, plan)
    return plan.with_estimates(cost=cost_est)


class DynamicProgrammingEnumerator(PlanEnumerator):
    """A very basic dynamic programming-based plan enumerator.

    This enumerator is very basic because it does not implement any sophisticated pruning rules or traversal strategies and
    only focuses on a small subset of possible operators. It simply enumerates all possible access paths and join paths and
    picks the cheapest one. This should only serve as a starting point when lacking an actual decent enumerator implementation
    (see *Limitation* below). Its purpose is mainly to shield users that are only interested in the cost model or the
    cardinality estimator from having to implement their own enumerator in order to use the `TextBookOptimizationPipeline`.
    Notice that for experiments based on PostgreSQL, a much more sophisticated implementation is available with the
    `PostgresDynProg` enumerator (and this enumerator is automatically selected when using the textbook pipeline with a
    Postgres target database).

    Limitations
    -----------

    - Only the cheapest access paths are considered, without taking sort orders into account. This prevents free merge join
      optimizations, i.e. if an access path is more expensive but already sorted, it will be discarded in favor of a cheaper
      alternative, even though a later merge join might become much cheaper due to the sort order.
    - No optimizations to intermediates are considered, i.e. no materialization or memoization of subplans.
    - Only the basic scan and join operators are considered. For scans, this includes sequential scan, index scan, index-only
      scan and bitmap scan. For joins, this includes nested loop join, hash join and sort merge join. These can be further
      restricted through the `supported_scan_ops` and `supported_join_ops` parameters.
    - Only simple SPJ queries are supported. Importantly, the query may not contain any set operations, subqueries, CTEs etc.
      All joins must be inner equijoins and no cross products are allowed.
    - Aggregations, sorting, etc. are not considered. In this way, the enumerator is comparable to the ``join_search_hook``
      of PostgreSQL. We assume that such "technicalities" are handled when creating appropriate hints for the target database
      or when executing the query on the target database at the latest.

    Parameters
    ----------
    supported_scan_ops : Optional[set[ScanOperator]], optional
        The set of scan operators that should be considered during the enumeration. This should be a subset of the following
        operators: sequential scan, index scan, index-only scan, bitmap scan. If any other operators are included, these
        are simply never considered. By default all operators that are available on the `target_db` are allowed.
    supported_join_ops : Optional[set[JoinOperator]], optional
        The set of join operators that should be considered during the enumeration. This should be a subset of the following
        operators: nested loop join, hash join, sort merge join. If any other operators are included, these are simply never
        considered. By default all operators that are available on the `target_db` are allowed.
    target_db : Optional[db.Database], optional
        The target database system for which the optimization pipeline is intended. If not omitted, the database is inferred
        from the `DatabasePool`.
    """

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

        dp_table = self._determine_base_access_paths(query, cost_model=cost_model,
                                                     cardinality_estimator=cardinality_estimator)
        final_plan = self._build_join_paths(query, dp_table=dp_table, cost_model=cost_model,
                                            cardinality_estimator=cardinality_estimator)

        cost_model.cleanup()
        cardinality_estimator.cleanup()
        return final_plan

    def pre_check(self) -> OptimizationPreCheck:
        return validation.merge_checks(
            validation.CrossProductPreCheck(),
            validation.VirtualTablesPreCheck(),
            validation.EquiJoinPreCheck(),
            validation.InnerJoinPreCheck(),
            validation.SubqueryPreCheck(),
            validation.SetOperationsPreCheck()
        )

    def describe(self) -> jsondict:
        return {
            "name": "dynamic_programming",
            "flavor": "default",
            "scan_ops": [op.name for op in self._scan_ops],
            "join_ops": [op.name for op in self._join_ops],
            "database_system": self._target_db.describe()
        }

    def _determine_base_access_paths(self, query: SqlQuery, *, cost_model: CostModel,
                                     cardinality_estimator: CardinalityEstimator) -> DPTable:
        """Initializes a new dynamic programming table which includes the cheapest access paths for each base table.

        The base tables are directly inferred from the query.
        """
        dp_table: DPTable = {}

        for table in query.tables():

            # We determine access paths in two phases: initially, we just gather all possible access paths to a specific table.
            # Aftewards, we evaluate these candidates according to our cost model and select the cheapest one.
            candidate_plans: list[QueryPlan] = []

            if ScanOperator.SequentialScan in self._scan_ops:
                candidate_plans.append(QueryPlan(ScanOperator.SequentialScan, base_table=table))
            candidate_plans += self._determine_index_paths(query, table)

            candidate_plans = [_calc_plan_estimates(query, candidate,
                                                    cost_model=cost_model, cardinality_estimator=cardinality_estimator)
                               for candidate in candidate_plans]

            cheapest_plan = min(candidate_plans, key=lambda plan: plan.estimated_cost)
            dp_table[frozenset([table])] = cheapest_plan

        return dp_table

    def _determine_index_paths(self, query: SqlQuery, table: TableReference) -> Iterable[QueryPlan]:
        """Gathers all possible index access paths for a specific table.

        The access paths do not contain a cost or cardinality estimates, yet. These information must be added by the caller.
        """
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
                    candidate_plans.append(QueryPlan(ScanOperator.IndexScan, base_table=table,
                                                     index=index, sort_keys=sorting))
                if can_idx_only_scan and ScanOperator.IndexOnlyScan in self._scan_ops:
                    candidate_plans.append(QueryPlan(ScanOperator.IndexOnlyScan, base_table=table,
                                                     index=index, sort_keys=sorting))

        if ScanOperator.BitmapScan in self._scan_ops:
            # The target DB/cost model is responsible for figuring out good bitmap index hierarchies.
            # Since bitmap scans combine multiple indexes, we do not consider bitmap scans in the above loop.
            # Furthermore, bitmap scans are partial sequential scans and thus do not provide a sort key.
            candidate_plans.append(QueryPlan(ScanOperator.BitmapScan, base_table=table, indexes=candidate_indexes))

        return candidate_plans

    def _build_join_paths(self, query: SqlQuery, *, dp_table: DPTable, cost_model: CostModel,
                          cardinality_estimator: CardinalityEstimator) -> QueryPlan:
        """Main optimization loop for the dynamic programmer.

        In this loop we construct increasingly large join paths by combining the optimal access paths of their input relations.
        At the end of the loop we have just constructed the cheapest join path for the entire query.

        All access paths are stored in the `dp_table`. This method assumes that the `dp_table` already contains the cheapest
        access paths for all base relations.

        Returns
        -------
        QueryPlan
            The final query plan that represents the cheapest join path for the given query.
        """

        predicates = query.predicates()
        candidate_tables = query.tables()

        for current_level in range(2, len(candidate_tables) + 1):

            # The current level describes how large the intermediate join paths that we are considering next are going to be.
            # For each potential intermediate that matches the current level, we determine the cheapest access path. This path
            # is going to re-use the cheapest access paths that we determined as part of an earlier iteration.

            current_intermediates = itertools.combinations(candidate_tables, current_level)
            access_paths = {
                frozenset(join): self._determine_cheapest_path(query, join, dp_table=dp_table, cost_model=cost_model,
                                                               cardinality_estimator=cardinality_estimator)
                for join in current_intermediates
                if predicates.joins_tables(join)  # we do not consider cross products
            }
            dp_table.update(access_paths)

        return dp_table[frozenset(candidate_tables)]

    def _determine_cheapest_path(self, query: SqlQuery, intermediate: Iterable[TableReference], *, dp_table: DPTable,
                                 cost_model: CostModel, cardinality_estimator: CardinalityEstimator) -> QueryPlan:
        """DP subroutine that selects the cheapest access path for a specific intermediate."""
        intermediate = frozenset(intermediate)
        candidate_plans: list[QueryPlan] = []

        # We determine the cheapest access path to our intermediate by checking all potential join partners that could possibly
        # be used to construct this intermediate. This works by splitting the intermediate into an outer relation and an inner
        # one. To guarantee that we test each possible split, we generate the entire power set of the intermediate.
        # By basing our algorithm on the power set we can solve two important problems: first, we can easily generate bushy
        # plans (each time the outer plan has at least two tables and leaves more than one table for the inner plan results in
        # a bushy plan). Second, we can also generate plans that invert the role of inner and outer relation just as easy.
        # This is because the power set will eventually visit the set of all tables in the (former) inner relation, which will
        # then become the outer relation for the current iteration.
        #
        # Once again, we first gather all possible join paths and then evaluate the costs for each of them in order to select
        # the cheapest one.

        for outer in util.collections.powerset(intermediate):
            if not outer or len(outer) == len(intermediate):
                # Skip the empty set and the full set because we would lack a join partner.
                continue
            outer = frozenset(outer)

            # All tables of our intermediate that are not part of the outer relation have to become part of the inner relation
            inner = intermediate - outer

            outer_plan, inner_plan = dp_table.get(outer), dp_table.get(inner)
            if not outer_plan or not inner_plan:
                # If we do not find the access paths for one of our inputs, it means that this is constructed using a cross
                # product. Since we do not consider cross products, we can skip this split.
                continue

            if JoinOperator.NestedLoopJoin in self._join_ops:
                candidate_plans.append(QueryPlan(JoinOperator.NestedLoopJoin, children=[outer_plan, inner_plan]))

            if JoinOperator.HashJoin in self._join_ops:
                candidate_plans.append(QueryPlan(JoinOperator.HashJoin, children=[outer_plan, inner_plan]))

            if JoinOperator.SortMergeJoin in self._join_ops:
                # The target DB is utimately responsible for figuring out whether it needs explicit sorts or whether it can
                # just merge directly.
                candidate_plans.append(QueryPlan(JoinOperator.SortMergeJoin, children=[outer_plan, inner_plan]))

        candidate_plans = [
            _calc_plan_estimates(query, candidate, cost_model=cost_model, cardinality_estimator=cardinality_estimator)
            for candidate in candidate_plans
        ]

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
