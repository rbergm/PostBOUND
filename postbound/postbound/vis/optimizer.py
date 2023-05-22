"""Utilities to visualize different aspects of query optimization, namely join trees and join graphs."""
from __future__ import annotations

from collections.abc import Sequence

import graphviz as gv
import networkx as nx

from postbound.qal import qal
from postbound.optimizer import joingraph, jointree
from postbound.vis import trees as tree_viz


def _join_tree_labels(node: jointree.AbstractJoinTreeNode) -> tuple[str, dict]:
    if node.is_join_node():
        return "⋈", {"style": "bold"}
    assert isinstance(node, jointree.BaseTableNode)
    return node.table.full_name, {"color": "grey"}


def _join_tree_traversal(node: jointree.AbstractJoinTreeNode) -> Sequence[jointree.AbstractJoinTreeNode]:
    if node.is_base_table_node():
        return ()
    assert isinstance(node, jointree.IntermediateJoinNode)
    return node.children


def plot_join_tree(join_tree: jointree.JoinTree) -> gv.Graph:
    if not join_tree:
        return gv.Graph()
    return tree_viz.plot_tree(join_tree.root, _join_tree_labels, _join_tree_traversal)


def _plot_join_graph_from_query(query: qal.SqlQuery) -> gv.Graph:
    if not query.predicates():
        return gv.Graph()
    join_graph: nx.Graph = query.predicates().join_graph()
    gv_graph = gv.Graph()
    for table in join_graph.nodes:
        gv_graph.node(str(table))
    for start, target in join_graph.edges:
        gv_graph.edge(str(start), str(target))
    return gv_graph


def _plot_join_graph_directly(join_graph: joingraph.JoinGraph) -> gv.Digraph:
    gv_graph = gv.Digraph()
    for table in join_graph:
        node_color = "black" if join_graph.is_free_table(table) else "blue"
        gv_graph.node(str(table), color=node_color)
    for start, target in join_graph.all_joins():
        if join_graph.is_pk_fk_join(start, target):
            gv_graph.edge(str(target), str(start))
        elif join_graph.is_pk_fk_join(target, start):
            gv_graph.edge(str(start), str(target))
        else:
            gv_graph.edge(str(start), str(target), dir="none")
    return gv_graph


def plot_join_graph(query_or_join_graph: qal.SqlQuery | joingraph.JoinGraph) -> gv.Graph | gv.Digraph:
    if isinstance(query_or_join_graph, qal.SqlQuery):
        return _plot_join_graph_from_query(query_or_join_graph)
    elif isinstance(query_or_join_graph, joingraph.JoinGraph):
        return _plot_join_graph_directly(query_or_join_graph)
    else:
        raise TypeError("Argument must be either SqlQuery or JoinGraph, not" + str(type(query_or_join_graph)))
