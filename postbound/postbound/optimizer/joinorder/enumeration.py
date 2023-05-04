"""Contains the abstractions for the join order optimization."""
from __future__ import annotations

import abc
import functools
import operator
import copy
import random
from collections.abc import Sequence
from typing import Iterable, Optional, Generator

import networkx as nx
import numpy as np

from postbound.qal import qal, base, predicates as preds
from postbound.db import db
from postbound.optimizer import data, validation
from postbound.optimizer.bounds import joins as join_bounds, scans as scan_bounds, stats
from postbound.optimizer.joinorder import subqueries
from postbound.util import networkx as nx_utils


class JoinOrderOptimizer(abc.ABC):
    """The `JoinOrderOptimizer` handles the entire process of obtaining a join order for input queries.

    The join ordering is the first step in the optimization process. Therefore, the implemented optimization strategy
    can apply an entire green-field approach.
    """

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.SqlQuery) -> data.JoinTree | None:
        """Performs the actual join ordering process.

        If for some reason there is no valid join order for the given query (e.g. queries with just a single selected
        table), `None` can be returned. Otherwise, the selected join order has to be described using a `JoinTree`.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy (e.g. for integrated optimization algorithms that are used in many
        real-world systems).

        Other than the join order and operator assignment, the algorithm should add as much information to the join
        tree as possible, e.g. including join conditions and cardinality estimates that were calculated for the
        selected joins. This ensures that other parts of the code base work as expected.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the join order optimization strategy."""
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly."""
        return validation.EmptyPreCheck()


@functools.cache
def _fetch_filters(query: qal.SqlQuery, table: base.TableReference) -> preds.AbstractPredicate | None:
    """Merges all filters for the given table into one predicate."""
    all_filters = query.predicates().filters_for(table)
    predicate = preds.CompoundPredicate.create_and(all_filters) if all_filters else None
    return predicate


@functools.cache
def _fetch_joins(query: qal.SqlQuery, joined_tables: base.TableReference | frozenset[base.TableReference],
                 intermediate_tables: base.TableReference | frozenset[base.TableReference]
                 ) -> Optional[preds.AbstractPredicate]:
    """Provides all join predicates between the given sets of tables.

    Join predicates between tables within each set are not considered. If no predicates are found, `None` is returned.
    Otherwise, the predicate will be the conjunction of all individual predicates.

    Theoretically speaking, the joined tables are intended as a new set of tables that have not been included in an
    intermediate join result of the input query, whereas the intermediate tables are supposed to form the intermediate.
    However, this is never actually enforced.
    """
    predicates = query.predicates()
    joins = []
    for joined_table in frozenset(joined_tables):
        for intermediate_table in frozenset(intermediate_tables):
            current_predicates = predicates.joins_between(joined_table, intermediate_table)
            if not current_predicates:
                continue
            joins.append(current_predicates)

    if not joins:
        return None
    return preds.CompoundPredicate.create_and(joins)


class UESJoinOrderOptimizer(JoinOrderOptimizer):
    """Implementation of the UES join order algorithm.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for a formal introduction into the
    algorithm

    The actual implementation used here expands upon the original algorithm in several ways:

    - conjunctive join predicates are supported by using the smallest bound of all components of the conjunction
    - queries with cross products are optimized on a per-connected-component basis. The final join order is created
    such that components with smaller upper bounds are joined first
    - star queries that only consist of primary key/foreign key joins are optimized using a derived algorithm:
    the smallest bound pk/fk join is used as the first join. Afterwards, the smallest bound tables are inserted
    greedily
    - paths of pk/fk joins are still inserted greedily, but their subtree is iterated using a BFS based on the base
    table estimates
    """

    def __init__(self, *, base_table_estimation: scan_bounds.BaseTableCardinalityEstimator,
                 join_estimation: join_bounds.JoinBoundCardinalityEstimator,
                 subquery_policy: subqueries.SubqueryGenerationPolicy,
                 stats_container: stats.StatisticsContainer,
                 database: db.Database, verbose: bool = False) -> None:
        super().__init__("UES enumeration")
        self.base_table_estimation = base_table_estimation
        self.join_estimation = join_estimation
        self.subquery_policy = subquery_policy
        self.stats_container = stats_container
        self.database = database
        self._logging_enabled = verbose

    def optimize_join_order(self, query: qal.SqlQuery) -> data.JoinTree | None:
        if len(query.tables()) < 2:
            return None

        self.base_table_estimation.setup_for_query(query)
        self.stats_container.setup_for_query(query, self.base_table_estimation)
        self.join_estimation.setup_for_query(query, self.stats_container)
        self.subquery_policy.setup_for_query(query, self.stats_container)

        join_graph = data.JoinGraph(query, self.database.schema())

        if len(query.tables()) == 2:
            final_join_tree = self._binary_join_optimization(query, join_graph)
        elif join_graph.contains_cross_products():
            # cross-product query is reduced to multiple independent optimization passes
            optimized_components = []
            for component in join_graph.join_components():
                # FIXME: join components might consist of single tables!
                optimized_component = self._clone().optimize_join_order(component.query)
                if not optimized_component:
                    raise JoinOrderOptimizationError(component.query)
                optimized_components.append(optimized_component)

            # insert cross-products such that the smaller partitions are joined first
            sorted(optimized_components, key=operator.attrgetter("upper_bound"))
            final_join_tree = data.JoinTree.cross_product_of(*optimized_components)
        elif join_graph.contains_free_n_m_joins():
            final_join_tree = self._default_ues_optimizer(query, join_graph)
        else:
            final_join_tree = self._star_query_optimizer(query, join_graph)

        return final_join_tree

    def describe(self) -> dict:
        return {
            "name": "ues",
            "settings": {
                "base_table_estimation": self.base_table_estimation.describe(),
                "join_estimation": self.join_estimation.describe(),
                "subqueries": self.subquery_policy.describe(),
                "statistics": self.stats_container.describe()
            }
        }

    def pre_check(self) -> validation.OptimizationPreCheck:
        specified_checks = [check for check in [self.base_table_estimation.pre_check(),
                                                self.join_estimation.pre_check(),
                                                self.subquery_policy.pre_check()]
                            if check]
        specified_checks.append(validation.UESOptimizationPreCheck())
        return validation.merge_checks(specified_checks)

    def _default_ues_optimizer(self, query: qal.SqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        """Implementation of our take on the UES algorithm for n:m joined queries."""
        join_tree = data.JoinTree()

        while join_graph.contains_free_n_m_joins():

            # Update the current upper bounds
            lowest_bound = np.inf
            lowest_bound_table = None
            for candidate_join in join_graph.available_n_m_join_paths():
                candidate_table = candidate_join.target_table
                filter_estimate = self.stats_container.base_table_estimates[candidate_table]
                pk_fk_bounds = [self.join_estimation.estimate_for(join_path.join_condition, join_graph) for join_path
                                in join_graph.available_pk_fk_joins_for(candidate_table)]
                candidate_min_bound = min([filter_estimate] + pk_fk_bounds)
                self.stats_container.upper_bounds[candidate_table] = candidate_min_bound

                if candidate_min_bound < lowest_bound:
                    lowest_bound = candidate_min_bound
                    lowest_bound_table = candidate_table

            if join_tree.is_empty():
                filter_pred = _fetch_filters(query, lowest_bound_table)
                join_tree = data.JoinTree.for_base_table(lowest_bound_table, lowest_bound, filter_pred)
                join_graph.mark_joined(lowest_bound_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                pk_joins = join_graph.available_deep_pk_join_paths_for(lowest_bound_table,
                                                                       self._table_base_cardinality_ordering)
                for pk_join in pk_joins:
                    target_table = pk_join.target_table
                    base_cardinality = self.stats_container.base_table_estimates[target_table]
                    filter_pred = _fetch_filters(query, target_table)
                    join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
                    join_graph.mark_joined(target_table, pk_join.join_condition)
                    join_tree = join_tree.join_with_base_table(pk_join.target_table, base_cardinality=base_cardinality,
                                                               base_filter_predicate=filter_pred,
                                                               join_predicate=pk_join.join_condition,
                                                               join_bound=join_bound, n_m_join=False)
                self._log_optimization_progress("Initial table selection", lowest_bound_table, pk_joins)
                continue

            selected_candidate: data.JoinPath | None = None
            lowest_bound = np.inf
            for candidate_join in join_graph.available_join_paths():
                candidate_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
                if candidate_bound < lowest_bound:
                    selected_candidate = candidate_join
                    lowest_bound = candidate_bound

            direct_pk_joins = join_graph.available_pk_fk_joins_for(selected_candidate.target_table)
            create_subquery = any(self.subquery_policy.generate_subquery_for(pk_join.join_condition, join_graph)
                                  for pk_join in direct_pk_joins)
            candidate_table = selected_candidate.target_table
            all_pk_joins = join_graph.available_deep_pk_join_paths_for(candidate_table)
            candidate_filters = _fetch_filters(query, candidate_table)
            candidate_base_cardinality = self.stats_container.base_table_estimates[candidate_table]
            self._log_optimization_progress("n:m join", candidate_table, all_pk_joins,
                                            join_condition=selected_candidate.join_condition,
                                            subquery_join=create_subquery)
            if create_subquery:
                subquery_tree = data.JoinTree.for_base_table(candidate_table, candidate_base_cardinality,
                                                             candidate_filters)
                join_graph.mark_joined(candidate_table)
                self._insert_pk_joins(query, all_pk_joins, subquery_tree, join_graph)
                join_tree = join_tree.join_with_subquery(subquery_tree, selected_candidate.join_condition, lowest_bound,
                                                         n_m_table=candidate_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
            else:
                join_tree = join_tree.join_with_base_table(candidate_table, base_cardinality=candidate_base_cardinality,
                                                           join_predicate=selected_candidate.join_condition,
                                                           join_bound=lowest_bound,
                                                           base_filter_predicate=candidate_filters)
                join_graph.mark_joined(candidate_table, selected_candidate.join_condition)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                join_tree = self._insert_pk_joins(query, all_pk_joins, join_tree, join_graph)

        if join_graph.contains_free_tables():
            raise AssertionError("Join graph still has free tables remaining!")
        return join_tree

    def _binary_join_optimization(self, query: qal.ImplicitSqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        """Simplified "optimization" for a query with a single join. Makes the smaller table the outer relation."""
        table1, table2 = query.tables()
        table1_smaller = self.stats_container.base_table_estimates[table1] < self.stats_container.base_table_estimates[
            table2]
        small_table, large_table = (table1, table2) if table1_smaller else (table2, table1)

        large_card = self.stats_container.base_table_estimates[large_table]
        small_card = self.stats_container.base_table_estimates[small_table]

        large_filter = _fetch_filters(query, large_table)
        small_filter = _fetch_filters(query, small_table)

        join_predicate = query.predicates().joins_between(large_table, small_table)
        join_bound = self.join_estimation.estimate_for(join_predicate, join_graph)

        join_tree = data.JoinTree.for_base_table(large_table, large_card, large_filter)
        join_tree = join_tree.join_with_base_table(small_table, base_cardinality=small_card,
                                                   base_filter_predicate=small_filter,
                                                   join_predicate=join_predicate, join_bound=join_bound,
                                                   n_m_join=join_graph.is_n_m_join(small_table, large_table))
        return join_tree

    def _star_query_optimizer(self, query: qal.ImplicitSqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        """UES-inspired algorithm for star queries (i.e. queries with pk/fk joins only)"""
        # initial table / join selection
        lowest_bound = np.inf
        lowest_bound_join = None
        for candidate_join in join_graph.available_join_paths():
            current_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
            if current_bound < lowest_bound:
                lowest_bound = current_bound
                lowest_bound_join = candidate_join

        start_table = lowest_bound_join.start_table
        start_filters = _fetch_filters(query, start_table)
        join_tree = data.JoinTree.for_base_table(start_table, self.stats_container.base_table_estimates[start_table],
                                                 start_filters)
        join_graph.mark_joined(start_table)
        join_tree = self._apply_pk_fk_join(query, lowest_bound_join, join_bound=lowest_bound, join_graph=join_graph,
                                           current_join_tree=join_tree)

        # join partner selection
        while join_graph.contains_free_tables():
            lowest_bound = np.inf
            lowest_bound_join = None
            for candidate_join in join_graph.available_join_paths():
                current_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
                if current_bound < lowest_bound:
                    lowest_bound = current_bound
                    lowest_bound_join = candidate_join

            join_tree = self._apply_pk_fk_join(query, lowest_bound_join, join_bound=lowest_bound, join_graph=join_graph,
                                               current_join_tree=join_tree)

        return join_tree

    def _table_base_cardinality_ordering(self, table: base.TableReference, join_edge: dict) -> int:
        """Utility method to enable an ordering of base tables according to their base table estimates."""
        return self.stats_container.base_table_estimates[table]

    def _apply_pk_fk_join(self, query: qal.SqlQuery, pk_fk_join: data.JoinPath, *, join_bound: int,
                          join_graph: data.JoinGraph, current_join_tree: data.JoinTree) -> data.JoinTree:
        """Includes the given  pk/fk join into the join tree, taking care of all necessary updates."""
        target_table = pk_fk_join.target_table
        target_filters = _fetch_filters(query, target_table)
        target_cardinality = self.stats_container.base_table_estimates[target_table]
        updated_join_tree = current_join_tree.join_with_base_table(target_table,
                                                                   join_predicate=pk_fk_join.join_condition,
                                                                   base_cardinality=target_cardinality,
                                                                   join_bound=join_bound,
                                                                   n_m_join=False,
                                                                   base_filter_predicate=target_filters)
        join_graph.mark_joined(target_table, pk_fk_join.join_condition)
        self.stats_container.upper_bounds[updated_join_tree] = join_bound
        return updated_join_tree

    def _insert_pk_joins(self, query: qal.SqlQuery, pk_joins: Iterable[data.JoinPath],
                         join_tree: data.JoinTree, join_graph: data.JoinGraph) -> data.JoinTree:
        """Generalization of `_apply_pk_fk_join` to multiple join paths."""
        # TODO: refactor in terms of _apply_pk_fk_join
        for pk_join in pk_joins:
            pk_table = pk_join.target_table
            if not join_graph.is_free_table(pk_table):
                continue
            pk_filters = _fetch_filters(query, pk_table)
            pk_join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
            pk_base_cardinality = self.stats_container.base_table_estimates[pk_table]
            join_tree = join_tree.join_with_base_table(pk_table, base_cardinality=pk_base_cardinality,
                                                       join_predicate=pk_join.join_condition,
                                                       join_bound=pk_join_bound,
                                                       base_filter_predicate=pk_filters,
                                                       n_m_join=False)
            join_graph.mark_joined(pk_table, pk_join.join_condition)
            self.stats_container.upper_bounds[join_tree] = pk_join_bound
        return join_tree

    def _clone(self) -> UESJoinOrderOptimizer:
        """Creates a new join order optimizer with the same settings as this one."""
        return UESJoinOrderOptimizer(base_table_estimation=copy.copy(self.base_table_estimation),
                                     join_estimation=copy.copy(self.join_estimation),
                                     subquery_policy=copy.copy(self.subquery_policy),
                                     stats_container=copy.copy(self.stats_container),
                                     database=self.database)

    def _log_optimization_progress(self, phase: str, candidate_table: base.TableReference,
                                   pk_joins: Iterable[data.JoinPath], *,
                                   join_condition: preds.AbstractPredicate | None = None,
                                   subquery_join: bool | None = None) -> None:
        """Logs the current optimization state."""
        # TODO: use proper logging
        if not self._logging_enabled:
            return
        log_components = [phase, "::", str(candidate_table), "with PK joins", str(pk_joins)]
        if join_condition:
            log_components.extend(["on condition", str(join_condition)])
        if subquery_join is not None:
            log_components.append("with subquery" if subquery_join else "without subquery")
        log_message = " ".join(log_components)
        print(log_message)


def _sample_join_path_snippet(query: qal.SqlQuery, join_snippet: Sequence[base.TableReference], *,
                              may_contain_cross_products: bool = False) -> Generator[data.JoinTree]:
    """Generates a random join path for the given excerpt of the query's tables.

    The join order only includes cross products if they are already contained in the input query.

    `may_contain_cross_products` indicates whether the input query contains any cross-products and speeds up the check
    whether a join order is valid significantly for queries that do not contain cross products.
    """
    if not join_snippet:
        raise ValueError("Empty join snippet")

    if len(join_snippet) == 1:
        base_table = join_snippet[0]
        filter_predicate = _fetch_filters(query, base_table)
        yield data.JoinTree.for_base_table(base_table, np.nan, filter_predicate)
        return
    elif len(join_snippet) == 2:
        base_table, joined_table = join_snippet
        if random.random() < 0.5:
            base_table, joined_table = joined_table, base_table
        join_predicate = query.predicates().joins_between(base_table, joined_table)
        if not may_contain_cross_products and not join_predicate:
            return

        base_filter = _fetch_filters(query, base_table)
        join_tree = data.JoinTree.for_base_table(base_table, np.nan, base_filter)

        joined_filter = _fetch_filters(query, joined_table)
        join_tree = join_tree.join_with_base_table(joined_table, base_cardinality=np.nan,
                                                   join_predicate=join_predicate,
                                                   join_bound=np.nan, base_filter_predicate=joined_filter)
        yield join_tree
        return

    available_splits = list(range(1, len(join_snippet)))
    while available_splits:
        split_idx = random.choice(available_splits)
        available_splits.remove(split_idx)

        head_tables = join_snippet[:split_idx]
        tail_tables = join_snippet[split_idx:]
        join_predicate = _fetch_joins(query, frozenset(head_tables), frozenset(tail_tables))
        if not may_contain_cross_products and not join_predicate:
            continue

        for head_tree in _sample_join_path_snippet(query, head_tables,
                                                   may_contain_cross_products=may_contain_cross_products):
            for tail_tree in _sample_join_path_snippet(query, tail_tables,
                                                       may_contain_cross_products=may_contain_cross_products):
                join_tree = data.JoinTree.joining(tail_tree, head_tree, join_condition=join_predicate)
                yield join_tree


def _sample_join_path(query: qal.SqlQuery,
                      linear_join_path: nx_utils.GraphWalk | Sequence[base.TableReference]) -> Generator[data.JoinTree]:
    """Provides a random (potentially bushy) join path based on the given linear join path.

    The join order only includes cross products if they are already contained in the input query.
    """
    n_cross_products = len(list(nx.connected_components(query.predicates().join_graph()))) - 1
    nodes = linear_join_path.nodes() if isinstance(linear_join_path, nx_utils.GraphWalk) else linear_join_path
    for join_tree in _sample_join_path_snippet(query, nodes, may_contain_cross_products=n_cross_products > 0):
        if join_tree.count_cross_product_joins() > n_cross_products:
            continue
        yield join_tree


class RandomJoinOrderGenerator:
    """Utility service that provides access to a generator to produce random join orders."""

    def random_join_order_generator(self, query: qal.SqlQuery) -> Generator[data.JoinTree]:
        """Generator that produces a new random (bushy) join order for the given input query upon each next-call.

        The join order only includes cross products if they are already contained in the input query.
        """
        while True:
            join_permutation = [tab for tab in nx_utils.nx_random_walk(query.predicates().join_graph())]
            try:
                join_order = next(_sample_join_path(query, join_permutation))
                yield join_order
            except StopIteration:
                continue


class RandomJoinOrderOptimizer(JoinOrderOptimizer):
    """Optimizer that generates a (not entirely) random join order.

    The join order only introduces cross-products, if those are necessary based on the input query.
    """

    def __init__(self) -> None:
        super().__init__("Random enumeration")
        self._generator = RandomJoinOrderGenerator()

    def optimize_join_order(self, query: qal.SqlQuery) -> data.JoinTree | None:
        return next(self._generator.random_join_order_generator(query))

    def describe(self) -> dict:
        return {"name": "random"}


def _all_join_tree_snippets(query: qal.SqlQuery, join_sequence: Sequence[base.TableReference], *,
                            may_contain_cross_products: bool = False) -> Generator[data.JoinTree]:
    """Provides all possible (potentially bushy) join trees that can be derived from the given tables.

    The join order only includes cross products if they are already contained in the input query.

    `may_contain_cross_products` indicates whether the input query contains any cross-products and speeds up the check
    whether a join order is valid significantly for queries that do not contain cross products.
    """
    if not join_sequence:
        raise ValueError("Empty join sequence")

    if len(join_sequence) == 1:
        base_table = join_sequence[0]
        filter_predicate = _fetch_filters(query, base_table)
        yield data.JoinTree.for_base_table(base_table, np.nan, filter_predicate)
        return
    elif len(join_sequence) == 2:
        base_table, joined_table = join_sequence
        join_predicate = query.predicates().joins_between(base_table, joined_table)
        if not may_contain_cross_products and not join_predicate:
            return

        base_filter = _fetch_filters(query, base_table)
        join_tree = data.JoinTree.for_base_table(base_table, np.nan, base_filter)

        joined_filter = _fetch_filters(query, joined_table)
        join_tree = join_tree.join_with_base_table(joined_table, base_cardinality=np.nan, join_predicate=join_predicate,
                                                   join_bound=np.nan, base_filter_predicate=joined_filter)
        yield join_tree
        return

    for split_idx in range(1, len(join_sequence)):
        head_tables = join_sequence[:split_idx]
        tail_tables = join_sequence[split_idx:]
        join_predicate = _fetch_joins(query, frozenset(head_tables), frozenset(tail_tables))
        if not may_contain_cross_products and not join_predicate:
            continue
        for head_tree in _all_join_tree_snippets(query, head_tables,
                                                 may_contain_cross_products=may_contain_cross_products):
            for tail_tree in _all_join_tree_snippets(query, tail_tables,
                                                     may_contain_cross_products=may_contain_cross_products):
                join_tree = data.JoinTree.joining(tail_tree, head_tree, join_condition=join_predicate)
                yield join_tree


def _all_branching_join_paths(query: qal.SqlQuery, linear_join_path: nx_utils.GraphWalk | Sequence[base.TableReference]
                              ) -> Generator[data.JoinTree]:
    """Generates all possible (bushy) join trees from the given linear join path.

    The join order only includes cross products if they are already contained in the input query.
    """
    n_cross_products = len(list(nx.connected_components(query.predicates().join_graph()))) - 1
    nodes = linear_join_path.nodes() if isinstance(linear_join_path, nx_utils.GraphWalk) else linear_join_path
    for join_tree in _all_join_tree_snippets(query, nodes, may_contain_cross_products=n_cross_products > 0):
        if join_tree.count_cross_product_joins() > n_cross_products:
            continue
        yield join_tree


class ExhaustiveJoinOrderGenerator:
    """Utility service that provides access to a generator that systematically produces all possible join trees.

    The join trees can be bushy and only include cross products if they are already specified in the input query.
    """

    def all_join_orders_for(self, query: qal.SqlQuery) -> Generator[data.JoinTree]:
        """Generator that yields all possible join orders for the input query.

        The join trees can be bushy and only include cross products if they are already specified in the input query.
        """
        linear_join_path_hashes = set()
        for linear_join_sequence in nx_utils.nx_frontier_walks(query.predicates().join_graph()):
            linear_join_sequence = tuple(linear_join_sequence.nodes())

            current_hash = hash(linear_join_sequence)
            if current_hash in linear_join_path_hashes:
                continue
            linear_join_path_hashes.add(current_hash)

            yield from _all_branching_join_paths(query, linear_join_sequence)


class EmptyJoinOrderOptimizer(JoinOrderOptimizer):
    """Dummy implementation of the join order optimizer that does not actually optimize anything."""

    def __init__(self) -> None:
        super().__init__("empty")

    def optimize_join_order(self, query: qal.SqlQuery) -> data.JoinTree | None:
        return None

    def describe(self) -> dict:
        return {"name": "no_ordering"}


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order."""

    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query
