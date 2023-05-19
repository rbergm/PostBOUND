"""Contains the abstractions for the join order optimization."""
from __future__ import annotations

import abc
import functools
import random
from collections.abc import Sequence
from typing import Optional, Generator

import networkx as nx

from postbound.qal import qal, base, predicates as preds
from postbound.optimizer import jointree, validation
from postbound.util import networkx as nx_utils


class JoinOrderOptimizer(abc.ABC):
    """The `JoinOrderOptimizer` handles the entire process of obtaining a join order for input queries.

    The join ordering is the first step in the optimization process. Therefore, the implemented optimization strategy
    can apply an entire green-field approach.
    """

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.SqlQuery
                            ) -> Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]:
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


def _sample_join_path_snippet(query: qal.SqlQuery, join_snippet: Sequence[base.TableReference], *,
                              may_contain_cross_products: bool = False) -> Generator[jointree.LogicalJoinTree]:
    """Generates a random join path for the given excerpt of the query's tables.

    The join order only includes cross products if they are already contained in the input query.

    `may_contain_cross_products` indicates whether the input query contains any cross-products and speeds up the check
    whether a join order is valid significantly for queries that do not contain cross products.
    """
    if not join_snippet:
        raise ValueError("Empty join snippet")

    if len(join_snippet) == 1:
        base_table = join_snippet[0]
        base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
        yield jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
        return
    elif len(join_snippet) == 2:
        base_table, joined_table = join_snippet
        if random.random() < 0.5:
            base_table, joined_table = joined_table, base_table
        join_predicate = query.predicates().joins_between(base_table, joined_table)
        if not may_contain_cross_products and not join_predicate:
            return

        base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
        base_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
        joined_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(joined_table))
        join_annotation = jointree.LogicalJoinMetadata(query.predicates().joins_between(base_table, joined_table))
        join_tree = base_tree.join_with_base_table(joined_table, joined_annotation, join_annotation)
        yield join_tree
        return

    available_splits = list(range(1, len(join_snippet)))
    while available_splits:
        split_idx = random.choice(available_splits)
        available_splits.remove(split_idx)

        head_tables = join_snippet[:split_idx]
        tail_tables = join_snippet[split_idx:]
        join_predicate = query.predicates().joins_between(head_tables, tail_tables)
        if not may_contain_cross_products and not join_predicate:
            continue

        for head_tree in _sample_join_path_snippet(query, head_tables,
                                                   may_contain_cross_products=may_contain_cross_products):
            for tail_tree in _sample_join_path_snippet(query, tail_tables,
                                                       may_contain_cross_products=may_contain_cross_products):
                annotation = jointree.LogicalJoinMetadata(join_predicate)
                join_tree = jointree.LogicalJoinTree.joining(tail_tree, head_tree, annotation)
                yield join_tree


def _sample_join_path(query: qal.SqlQuery, linear_join_path: nx_utils.GraphWalk | Sequence[base.TableReference]
                      ) -> Generator[jointree.LogicalJoinTree]:
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

    def random_join_order_generator(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
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
        super().__init__()
        self._generator = RandomJoinOrderGenerator()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return next(self._generator.random_join_order_generator(query))

    def describe(self) -> dict:
        return {"name": "random"}


def _all_join_tree_snippets(query: qal.SqlQuery, join_sequence: Sequence[base.TableReference], *,
                            may_contain_cross_products: bool = False) -> Generator[jointree.LogicalJoinTree]:
    """Provides all possible (potentially bushy) join trees that can be derived from the given tables.

    The join order only includes cross products if they are already contained in the input query.

    `may_contain_cross_products` indicates whether the input query contains any cross-products and speeds up the check
    whether a join order is valid significantly for queries that do not contain cross products.
    """
    if not join_sequence:
        raise ValueError("Empty join sequence")

    if len(join_sequence) == 1:
        base_table = join_sequence[0]
        annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
        yield jointree.LogicalJoinTree.for_base_table(base_table, annotation)
        return
    elif len(join_sequence) == 2:
        base_table, joined_table = join_sequence
        join_predicate = query.predicates().joins_between(base_table, joined_table)
        if not may_contain_cross_products and not join_predicate:
            return

        base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
        base_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
        joined_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(joined_table))
        join_annotation = jointree.LogicalJoinMetadata(join_predicate)
        join_tree = base_tree.join_with_base_table(joined_table, joined_annotation, join_annotation)
        yield join_tree
        return

    for split_idx in range(1, len(join_sequence)):
        head_tables = join_sequence[:split_idx]
        tail_tables = join_sequence[split_idx:]
        join_predicate = query.predicates().joins_between(head_tables, tail_tables)
        if not may_contain_cross_products and not join_predicate:
            continue
        for head_tree in _all_join_tree_snippets(query, head_tables,
                                                 may_contain_cross_products=may_contain_cross_products):
            for tail_tree in _all_join_tree_snippets(query, tail_tables,
                                                     may_contain_cross_products=may_contain_cross_products):
                annotation = jointree.LogicalJoinMetadata(join_predicate)
                join_tree = jointree.LogicalJoinTree.joining(tail_tree, head_tree, annotation)
                yield join_tree


def _all_branching_join_paths(query: qal.SqlQuery, linear_join_path: nx_utils.GraphWalk | Sequence[base.TableReference]
                              ) -> Generator[jointree.LogicalJoinTree]:
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

    def all_join_orders_for(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
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
        super().__init__()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return None

    def describe(self) -> dict:
        return {"name": "no_ordering"}


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order."""

    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query
