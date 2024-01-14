"""Utilities to visualize different aspects of query optimization, namely join trees and join graphs."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import graphviz as gv
import networkx as nx

from postbound.db import db
from postbound.qal import qal, base, transform
from postbound.optimizer import joingraph, jointree
from postbound.vis import trees as tree_viz


def _join_tree_labels(node: jointree.AbstractJoinTreeNode) -> tuple[str, dict]:
    if node.is_join_node():
        base_text = "⋈"
        base_style = {"style": "bold"}
    else:
        assert isinstance(node, jointree.BaseTableNode)
        base_text = str(node.table)
        base_style = {"color": "grey"}

    if "operator" in dir(node.annotation):
        base_text += "\n" + node.annotation.operator.operator.value

    return base_text, base_style


def _join_tree_traversal(node: jointree.AbstractJoinTreeNode) -> Sequence[jointree.AbstractJoinTreeNode]:
    if node.is_base_table_node():
        return ()
    assert isinstance(node, jointree.IntermediateJoinNode)
    return node.children


def plot_join_tree(join_tree: jointree.JoinTree) -> gv.Graph:
    if not join_tree:
        return gv.Graph()
    return tree_viz.plot_tree(join_tree.root, _join_tree_labels, _join_tree_traversal)


def _fallback_default_join_edge(graph: gv.Digraph, join_table: base.TableReference,
                                partner_table: base.TableReference) -> None:
    graph.edge(str(join_table), str(partner_table), dir="none")


def _render_pk_fk_join_edge(graph: gv.Digraph, query: qal.SqlQuery,
                            join_table: base.TableReference, partner_table: base.TableReference) -> None:
    db_schema = db.DatabasePool.get_instance().current_database().schema()
    join_predicate = query.predicates().joins_between(join_table, partner_table)
    if not join_predicate:
        return _fallback_default_join_edge(graph, join_table, partner_table)

    join_columns = join_predicate.join_partners()
    if len(join_columns) != 1:
        return _fallback_default_join_edge(graph, join_table, partner_table)

    join_col, partner_col = list(join_columns)[0]
    if db_schema.is_primary_key(join_col) and db_schema.has_secondary_index(partner_col):
        graph.edge(str(partner_col.table), str(join_col.table))
    elif db_schema.is_primary_key(partner_col) and db_schema.has_secondary_index(join_col):
        graph.edge(str(join_col.table), str(partner_col.table))
    else:
        _fallback_default_join_edge(graph, join_table, partner_table)


def _plot_join_graph_from_query(query: qal.SqlQuery, table_annotations: Optional[Callable[[base.TableReference], str]] = None,
                                include_pk_fk_joins: bool = False) -> gv.Graph:
    if not query.predicates():
        return gv.Graph()
    join_graph: nx.Graph = query.predicates().join_graph()
    gv_graph = gv.Digraph() if include_pk_fk_joins else gv.Graph
    for table in join_graph.nodes:
        node_label = str(table)
        node_label += ("\n" + table_annotations(table)) if table_annotations is not None else ""
        gv_graph.node(str(table), label=node_label)
    for start, target in join_graph.edges:
        if include_pk_fk_joins:
            _render_pk_fk_join_edge(gv_graph, query, start, target)
        else:
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
                    table_annotations: Optional[Callable[[base.TableReference], str]] = None, *,
                    include_pk_fk_joins: bool = False, out_path: str = "", out_format: str = "svg") -> gv.Graph | gv.Digraph:
    if isinstance(query_or_join_graph, qal.SqlQuery):
        graph = _plot_join_graph_from_query(query_or_join_graph, table_annotations, include_pk_fk_joins)
    elif isinstance(query_or_join_graph, joingraph.JoinGraph):
        graph = _plot_join_graph_directly(query_or_join_graph, table_annotations)
    else:
        raise TypeError("Argument must be either SqlQuery or JoinGraph, not" + str(type(query_or_join_graph)))

    if out_path:
        graph.render(out_path, format=out_format, cleanup=True)
    return graph


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


def merged_annotation(*annotations) -> Callable[[base.TableReference], str]:
    def _merger(table: base.TableReference) -> str:
        return "\n".join(annotator(table) for annotator in annotations)
    return _merger


def setup_annotations(*annotations: Literal["estimated-cards", "filter-cards", "true-cards"], query: qal.SqlQuery,
                      database: Optional[db.Database] = None) -> Callable[[base.TableReference], str]:
    annotators = []
    for annotator in annotations:
        if annotator == "estimated-cards":
            annotators.append(functools.partial(estimated_cards, query=query, database=database))
        elif annotator == "filter-cards":
            annotators.append(functools.partial(annotate_filter_cards, query=query, database=database))
        elif annotator == "true-cards":
            annotators.append(functools.partial(annotate_cards, query=query, database=database))
        else:
            raise ValueError(f"Unknown annotator: '{annotator}'")

    if not annotators:
        raise ValueError("No annotator given")
    return merged_annotation(*annotators) if len(annotators) > 1 else annotators[0]
