"""Utilities to visualize different aspects of query optimization, namely join trees and join graphs."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv
import networkx as nx

from postbound.db import db
from postbound.qal import qal, base, transform
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


def _plot_join_graph_from_query(query: qal.SqlQuery,
                                table_annotations: Optional[Callable[[base.TableReference], str]] = None) -> gv.Graph:
    if not query.predicates():
        return gv.Graph()
    join_graph: nx.Graph = query.predicates().join_graph()
    gv_graph = gv.Graph()
    for table in join_graph.nodes:
        node_label = str(table)
        node_label += ("\n" + table_annotations(table)) if table_annotations is not None else ""
        gv_graph.node(str(table), label=node_label)
    for start, target in join_graph.edges:
        gv_graph.edge(str(start), str(target))
    return gv_graph


def _plot_join_graph_directly(join_graph: joingraph.JoinGraph,
                              table_annotations: Optional[Callable[[base.TableReference], str]] = None) -> gv.Digraph:
    gv_graph = gv.Digraph()
    for table in join_graph:
        node_color = "black" if join_graph.is_free_table(table) else "blue"
        node_label = str(table)
        node_label += ("\n" + table_annotations(table)) if table_annotations is not None else ""
        gv_graph.node(str(table), label=node_label, color=node_color)
    for start, target in join_graph.all_joins():
        if join_graph.is_pk_fk_join(start, target):  # start is FK, target is PK
            gv_graph.edge(str(start), str(target))  # edge arrow goes from start to target (i.e. FK to PK)
        elif join_graph.is_pk_fk_join(target, start):  # target is FK, start is PK
            gv_graph.edge(str(target), str(start))  # edge arrow goes form target to start (i.e. FK to PK)
        else:
            gv_graph.edge(str(start), str(target), dir="none")
    return gv_graph


def plot_join_graph(query_or_join_graph: qal.SqlQuery | joingraph.JoinGraph,
                    table_annotations: Optional[Callable[[base.TableReference], str]] = None) -> gv.Graph | gv.Digraph:
    if isinstance(query_or_join_graph, qal.SqlQuery):
        return _plot_join_graph_from_query(query_or_join_graph, table_annotations)
    elif isinstance(query_or_join_graph, joingraph.JoinGraph):
        return _plot_join_graph_directly(query_or_join_graph, table_annotations)
    else:
        raise TypeError("Argument must be either SqlQuery or JoinGraph, not" + str(type(query_or_join_graph)))


def estimated_cards(table: base.TableReference, *, query: qal.SqlQuery, database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    filter_query = transform.as_star_query(filter_query)
    card_est = database.optimizer().cardinality_estimate(filter_query)
    return f"[{card_est} rows estimated]"


def annotate_filter_cards(table: base.TableReference, *, query: qal.SqlQuery,
                          database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    count_query = transform.as_count_star_query(filter_query)
    card = database.execute_query(count_query, cache_enabled=True)
    return f"[{card} rows]"


def annotate_cards(table: base.TableReference, *, query: qal.SqlQuery, database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    count_query = transform.as_count_star_query(filter_query)
    filter_card = database.execute_query(count_query, cache_enabled=True)
    total_card = database.statistics().total_rows(table, emulated=True, cache_enabled=True)
    return f"|R| = {total_card} |σ(R)| = {filter_card}"
