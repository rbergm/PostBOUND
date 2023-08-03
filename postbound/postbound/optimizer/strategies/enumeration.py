"""Provides exhaustive optimization strategies that enumerate all possible plans."""
from __future__ import annotations

from collections.abc import Generator

import networkx as nx

from postbound.qal import base, qal
from postbound.optimizer import jointree


def _merge_nodes(start: jointree.LogicalJoinTree | base.TableReference,
                 end: jointree.LogicalJoinTree | base.TableReference) -> jointree.LogicalJoinTree:
    start = jointree.LogicalJoinTree.for_base_table(start) if isinstance(start, base.TableReference) else start
    end = jointree.LogicalJoinTree.for_base_table(end) if isinstance(end, base.TableReference) else end
    return start.join_with_subtree(end)


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
