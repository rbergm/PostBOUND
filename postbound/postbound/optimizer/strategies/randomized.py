"""Provides "optimization" strategies that generate random query plans."""
from __future__ import annotations

import random
from collections.abc import Generator
from typing import Optional

import networkx as nx

from postbound.qal import base, qal
from postbound.optimizer import jointree, stages, validation


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


def _sample_join_graph(join_graph: nx.Graph) -> jointree.LogicalJoinTree:
    """Generates a random join order for the given join graph.

    Parameters
    ----------
    join_graph : nx.Graph
        The join graph that should be "optimized". This should be a pure join graph as provided by the *qal* module.

    Returns
    -------
    jointree.LogicalJoinTree
        A random join order for the given join graph.

    Warnings
    --------
    This algorithm does not work for join graphs that contain cross products (i.e. multiple connected components).

    Notes
    -----
    This algorithm works in an iterative manner: At each step, two connected nodes are selected. For these nodes, a join is
    simulated. This is done by generating a join tree for the nodes and merging them into a single node for the join tree. The
    iteration stops as soon as the graph only consists of a single node. This node represents the join tree for the entire
    graph. Depending on the order in which the edges are selected, a different join tree is produced.
    """
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
    """Utility service to produce randomized join orders for an input query.

    The service produces a generator that in turn provides the join orders. This is done in the `random_join_order_generator`
    method. The provided join orders include linear, as well as bushy join trees.

    Parameters
    ----------
    eliminate_duplicates : bool, optional
        Whether repeated calls to the generator should be guaranteed to provide different join orders. Defaults to ``False``,
        which permits duplicates.

    Warnings
    --------
    For now, the underlying algorithm is limited to queries without cross-products.
    """

    def __init__(self, eliminate_duplicates: bool = False) -> None:
        self._eliminate_duplicates = eliminate_duplicates

    def random_join_order_generator(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
        """Provides a generator that successively provides join orders at random.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which the join orders should be generated

        Yields
        ------
        Generator[jointree.LogicalJoinTree]
            A generator that produces random join orders for the input query, including bushy trees. Depeding on the
            `eliminate_duplicates` attribute, these join orders are guaranteed to be unique.

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
    """Optimization stage that produces a randomized join order.

    Currently, the algorithm only supports queries without cross products. The returned join order can be linear, as well as
    bushy.
    """

    def __init__(self) -> None:
        super().__init__()
        self._generator = RandomJoinOrderGenerator()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return next(self._generator.random_join_order_generator(query))

    def describe(self) -> dict:
        return {"name": "random"}

    def pre_check(self) -> validation.OptimizationPreCheck:
        return validation.CrossProductPreCheck()
