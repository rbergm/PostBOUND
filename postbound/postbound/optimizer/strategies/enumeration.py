"""Enumerative optimization strategies provide all possible plans in an exhaustive manner.

These strategies do not make use of any statistics, etc. to generate "good" plans. Instead, they focus on the structure of the
plans to generate new plans.
"""
from __future__ import annotations

from collections.abc import Generator

import networkx as nx

from postbound.qal import base, qal
from postbound.optimizer import jointree


def _merge_nodes(start: jointree.LogicalJoinTree | base.TableReference,
                 end: jointree.LogicalJoinTree | base.TableReference) -> jointree.LogicalJoinTree:
    """Provides a join tree that combines two specific trees or tables.

    This is a shortcut method to merge arbitrary tables or trees without having to check whether a table-based or tree-based
    merge has to be performed.

    Parameters
    ----------
    start : jointree.LogicalJoinTree | base.TableReference
        The first tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.
    end : jointree.LogicalJoinTree | base.TableReference
        The second tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.

    Returns
    -------
    jointree.LogicalJoinTree
        A join tree combining the input trees. The `start` node will be the left node of the tree and the `end` node will be
        the right node.
    """
    start = jointree.LogicalJoinTree.for_base_table(start) if isinstance(start, base.TableReference) else start
    end = jointree.LogicalJoinTree.for_base_table(end) if isinstance(end, base.TableReference) else end
    return start.join_with_subtree(end)


def _enumerate_join_graph(join_graph: nx.Graph) -> Generator[jointree.JoinTree]:
    """Provides all possible join trees based on a join graph.

    Parameters
    ----------
    join_graph : nx.Graph
        The join graph that should be "optimized". Due to the recursive nature of the method, this graph is not limited to a
        pure join graph as provided by the *qal* module. Instead, it nodes can already by join trees.

    Yields
    ------
    Generator[jointree.JoinTree]
        A possible join tree of the join graph.

    Warnings
    --------
    This algorithm does not work for join graphs that contain cross products (i.e. multiple connected components).

    Notes
    -----
    This algorithm works in a recursive manner: At each step, two connected nodes are selected. For these nodes, a join is
    simulated. This is done by generating a join tree for the nodes and merging them into a single node for the join tree. The
    recursion stops as soon as the graph only consists of a single node. This node represents the join tree for the entire
    graph. Depending on the order in which the edges are selected, a different join tree is produced. The exhaustive nature of
    this algorithm guarantees that all possible orders are selected.
    """
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
    """Utility service to provide all possible join trees for an input query.

    The service produces a generator that in turn provides the join orders. This is done in the `all_join_orders_for` method.
    The provided join orders include linear, as well as bushy join trees.

    Warnings
    --------
    For now, the underlying algorithm is limited to queries without cross-products.
    """

    def all_join_orders_for(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
        """Produces a generator for all possible join trees of a query.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"

        Yields
        ------
        Generator[jointree.LogicalJoinTree]
            A generator that produces all possible join orders for the input query, including bushy trees.

        Raises
        ------
        ValueError
            If the query contains cross products.

        Warnings
        --------
        For now, the underlying algorithm is limited to queries without cross-products.
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
