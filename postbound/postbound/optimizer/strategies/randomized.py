"""Provides "optimization" strategies that generate random query plans."""
from __future__ import annotations

import random
from collections.abc import Generator
from typing import Optional

import networkx as nx

from postbound.qal import base, qal
from postbound.optimizer import jointree, stages


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
        join_tree = (_merge_nodes(start_node, target_node) if random.random() < 0.5
                     else _merge_nodes(target_node, start_node))

        join_graph = nx.contracted_nodes(join_graph, start_node, target_node, self_loops=False)
        join_graph = nx.relabel_nodes(join_graph, {start_node: join_tree})

    final_node = list(join_graph.nodes)[0]
    return final_node


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


class RandomJoinOrderOptimizer(stages.JoinOrderOptimization):
    """Optimizer that generates a (not entirely) random join order."""

    def __init__(self) -> None:
        super().__init__()
        self._generator = RandomJoinOrderGenerator()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return next(self._generator.random_join_order_generator(query))

    def describe(self) -> dict:
        return {"name": "random"}
