"""Contains the abstractions for the join order optimization."""
from __future__ import annotations

import abc
import random
from typing import Optional, Generator

import networkx as nx

from postbound.qal import qal, base
from postbound.optimizer import jointree, validation


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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


def _merge_nodes(start: jointree.LogicalJoinTree | base.TableReference,
                 end: jointree.LogicalJoinTree | base.TableReference) -> jointree.LogicalJoinTree:
    start = jointree.LogicalJoinTree.for_base_table(start) if isinstance(start, base.TableReference) else start
    end = jointree.LogicalJoinTree.for_base_table(end) if isinstance(end, base.TableReference) else end
    return start.join_with_subtree(end)


def _sample_join_graph(join_graph: nx.Graph) -> jointree.LogicalJoinTree:
    while len(join_graph.nodes) > 1:
        join_predicates = list(join_graph.edges)
        next_edge = random.choice(join_predicates)
        start_node, target_node = next_edge
        join_tree = _merge_nodes(start_node, target_node) if random.random() < 0.5 else _merge_nodes(target_node, start_node)

        join_graph = nx.contracted_nodes(join_graph, start_node, target_node, self_loops=False)
        join_graph = nx.relabel_nodes(join_graph, {start_node: join_tree})

    final_node = list(join_graph.nodes)[0]
    return final_node


def _enumerate_join_graph(join_graph: nx.Graph) -> Generator[jointree.JoinTree]:
    if len(join_graph.nodes) == 1:
        node = list(join_graph.nodes)[0]
        yield node
        return

    for edge in join_graph.edges:
        start_node, target_node = edge
        merged_graph = nx.contracted_nodes(join_graph, start_node, target_node, self_loops=False, copy=True)

        start_end_tree = _merge_nodes(start_node, target_node)
        start_end_graph = nx.relabel_nodes(merged_graph, {start_node: start_end_tree}, copy=True)
        yield from _enumerate_join_graph(start_end_graph)

        end_start_tree = _merge_nodes(target_node, start_node)
        end_start_graph = nx.relabel_nodes(merged_graph, {start_node: end_start_tree}, copy=True)
        yield from _enumerate_join_graph(end_start_graph)


class RandomJoinOrderGenerator:
    """Utility service that provides access to a generator to produce random join orders."""

    def __init__(self, eliminate_duplicates: bool = False) -> None:
        self._eliminate_duplicates = eliminate_duplicates

    def random_join_order_generator(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
        """Generator that produces a new random (bushy) join order for the given input query upon each next-call."""
        join_graph = query.predicates().join_graph()
        if len(join_graph.nodes) == 0:
            return
        elif len(join_graph.nodes) == 1:
            base_table = list(join_graph.nodes)[0]
            base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
            join_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
            while True:
                yield join_tree
        elif not nx.is_connected(join_graph):
            raise ValueError("Cross products are not yet supported for random join order generation!")

        join_order_hashes = set()
        while True:
            current_join_order = _sample_join_graph(join_graph)
            if self._eliminate_duplicates:
                current_hash = hash(current_join_order)
                if current_hash in join_order_hashes:
                    continue
                else:
                    join_order_hashes.add(current_hash)

            yield current_join_order


class RandomJoinOrderOptimizer(JoinOrderOptimizer):
    """Optimizer that generates a (not entirely) random join order."""

    def __init__(self) -> None:
        super().__init__()
        self._generator = RandomJoinOrderGenerator()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return next(self._generator.random_join_order_generator(query))

    def describe(self) -> dict:
        return {"name": "random"}


class ExhaustiveJoinOrderGenerator:
    """Utility service that provides access to a generator that systematically produces all possible join trees.

    The join trees can be bushy and only include cross products if they are already specified in the input query.
    """

    def all_join_orders_for(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
        """Generator that yields all possible join orders for the input query.

        The join trees can be bushy and only include cross products if they are already specified in the input query.
        """
        join_graph = query.predicates().join_graph()
        if len(join_graph.nodes) == 0:
            return
        elif len(join_graph.nodes) == 1:
            base_table = list(join_graph.nodes)[0]
            base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
            join_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
            yield join_tree
            return
        elif not nx.is_connected(join_graph):
            raise ValueError("Cross products are not yet supported for random join order generation!")

        join_order_hashes = set()
        join_order_generator = _enumerate_join_graph(join_graph)
        for join_order in join_order_generator:
            current_hash = hash(join_order)
            if current_hash in join_order_hashes:
                continue

            join_order_hashes.add(current_hash)
            yield join_order


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
