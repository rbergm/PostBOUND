"""Utilities to visualize different aspects of query optimization, namely join trees and join graphs."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import graphviz as gv
import networkx as nx

from . import trees
from .. import db, util
from ..qal import relalg, transform, SqlQuery
from ..optimizer import joingraph, JoinTree, LogicalJoinTree
from .._core import TableReference
from .._qep import QueryPlan


def _join_tree_labels(node: JoinTree) -> tuple[str, dict]:
    if node.is_join():
        base_text = "⋈"
        base_style = {"style": "bold"}
    else:
        assert node.is_scan()
        base_text = str(node.base_table)
        base_style = {"color": "grey"}

    if isinstance(node, LogicalJoinTree):
        base_text += f"\n Card = {node.cardinality}"

    return base_text, base_style


def _join_tree_traversal(node: JoinTree) -> Sequence[JoinTree]:
    return node.children


def plot_join_tree(join_tree: JoinTree) -> gv.Graph:
    if not join_tree:
        return gv.Graph()
    return trees.plot_tree(join_tree, _join_tree_labels, _join_tree_traversal)


def _fallback_default_join_edge(graph: gv.Digraph, join_table: TableReference,
                                partner_table: TableReference) -> None:
    graph.edge(str(join_table), str(partner_table), dir="none")


def _render_pk_fk_join_edge(graph: gv.Digraph, query: SqlQuery,
                            join_table: TableReference, partner_table: TableReference) -> None:
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


def _plot_join_graph_from_query(query: SqlQuery, table_annotations: Optional[Callable[[TableReference], str]] = None,
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
                              table_annotations: Optional[Callable[[TableReference], str]] = None) -> gv.Digraph:
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


def plot_join_graph(query_or_join_graph: SqlQuery | joingraph.JoinGraph,
                    table_annotations: Optional[Callable[[TableReference], str]] = None, *,
                    include_pk_fk_joins: bool = False, out_path: str = "", out_format: str = "svg") -> gv.Graph | gv.Digraph:
    if isinstance(query_or_join_graph, SqlQuery):
        graph = _plot_join_graph_from_query(query_or_join_graph, table_annotations, include_pk_fk_joins)
    elif isinstance(query_or_join_graph, joingraph.JoinGraph):
        graph = _plot_join_graph_directly(query_or_join_graph, table_annotations)
    else:
        raise TypeError("Argument must be either SqlQuery or JoinGraph, not" + str(type(query_or_join_graph)))

    if out_path:
        graph.render(out_path, format=out_format, cleanup=True)
    return graph


def estimated_cards(table: TableReference, *, query: SqlQuery, database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    filter_query = transform.as_star_query(filter_query)
    card_est = database.optimizer().cardinality_estimate(filter_query)
    return f"[{card_est} rows estimated]"


def annotate_filter_cards(table: TableReference, *, query: SqlQuery,
                          database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    count_query = transform.as_count_star_query(filter_query)
    card = database.execute_query(count_query, cache_enabled=True)
    return f"[{card} rows]"


def annotate_cards(table: TableReference, *, query: SqlQuery, database: Optional[db.Database] = None) -> str:
    database = database if database is not None else db.DatabasePool.get_instance().current_database()
    filter_query = transform.extract_query_fragment(query, [table])
    count_query = transform.as_count_star_query(filter_query)
    filter_card = database.execute_query(count_query, cache_enabled=True)
    total_card = database.statistics().total_rows(table, emulated=True, cache_enabled=True)
    return f"|R| = {total_card} |σ(R)| = {filter_card}"


def merged_annotation(*annotations) -> Callable[[TableReference], str]:
    def _merger(table: TableReference) -> str:
        return "\n".join(annotator(table) for annotator in annotations)
    return _merger


def setup_annotations(*annotations: Literal["estimated-cards", "filter-cards", "true-cards"], query: SqlQuery,
                      database: Optional[db.Database] = None) -> Callable[[TableReference], str]:
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


def _query_plan_labels(node: QueryPlan, *,
                       annotation_generator: Optional[Callable[[QueryPlan], str]],
                       subplan_target: str = "") -> tuple[str, dict]:
    if node.subplan:
        label, params = _query_plan_labels(node.subplan.root, annotation_generator=annotation_generator,
                                           subplan_target=node.subplan.target_name)
        label = f"<<SubPlan>> {subplan_target}\n{label}"
        params["style"] = "dashed"
    elif node.is_join:
        label, params = node.node_type, {"style": "bold"}
    elif node.is_scan:
        label, params = f"<<{node.node_type}>>\n{node.base_table}", {"color": "grey"}
    else:
        label, params = node.node_type, {"style": "dashed", "color": "grey"}

    if node.subplan:
        label = f"{label}\nSubplan: {node.subplan.target_name}"

    annotation = annotation_generator(node) if annotation_generator else ""
    label = f"{label}\n{annotation}" if annotation else label
    return label, params


def _query_plan_traversal(node: QueryPlan, *, skip_intermediates: bool = False) -> list[QueryPlan]:
    children = list(node.children)
    if node.subplan:
        children.append(node.subplan.root)

    if skip_intermediates:
        skipped = [_query_plan_traversal(child, skip_intermediates=True) if child.is_auxiliary() else [child]
                   for child in children]
        children = util.flatten(skipped)
    return children


def annotate_estimates(node: QueryPlan) -> str:
    return f"cost={node.estimated_cost} cardinality={node.estimated_cardinality}"


def plot_query_plan(plan: QueryPlan,
                    annotation_generator: Optional[Callable[[QueryPlan], str]] = None, *,
                    skip_intermediates: bool = False, **kwargs) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return trees.plot_tree(plan, functools.partial(_query_plan_labels, annotation_generator=annotation_generator),
                           functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates),
                           **kwargs)


def _explain_analyze_annotations(node: QueryPlan) -> str:
    card_row = f"[Rows expected={node.estimated_cardinality} actual={node.actual_cardinality}]"
    exec_time = round(node.execution_time, 4)
    runtime_row = f"[Exec time={exec_time}s]"
    return card_row + "\n" + runtime_row


def plot_analyze_plan(plan: QueryPlan, *, skip_intermediates: bool = False, **kwargs) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return trees.plot_tree(plan,
                           functools.partial(_query_plan_labels, annotation_generator=_explain_analyze_annotations),
                           functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates),
                           **kwargs)


def _escape_label(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _make_sub(text: str) -> str:
    return f"<sub><font point-size='10.0'>{_escape_label(text)}</font></sub>"


def _make_label(text: str) -> str:
    return f"<b>{text}</b>"


def _relalg_node_labels(node: relalg.RelNode) -> tuple[str, dict]:
    node_params = {}
    match node:
        case relalg.Projection():
            projection_targets = ", ".join(str(t) for t in node.columns)
            node_str = f"{_make_label('π')} {_make_sub(projection_targets)}"
        case relalg.Selection():
            predicate = str(node.predicate)
            node_str = f"{_make_label('σ')} {_make_sub(predicate)}"
        case relalg.ThetaJoin():
            predicate = str(node.predicate)
            node_str = f"{_make_label('⋈')} {_make_sub(predicate)}"
        case relalg.SemiJoin():
            predicate = str(node.predicate)
            node_str = f"{_make_label('⋉')} {_make_sub(predicate)}"
        case relalg.AntiJoin():
            predicate = str(node.predicate)
            node_str = f"{_make_label('▷')} {_make_sub(predicate)}"
        case relalg.GroupBy():
            columns_str = ", ".join(str(c) for c in node.group_columns)
            aggregates: list[str] = []
            for group_columns, agg_func in node.aggregates.items():
                if len(group_columns) == 1:
                    group_str = str(util.simplify(group_columns))
                else:
                    group_str = "(" + ", ".join(str(c) for c in group_columns) + ")"

                if len(agg_func) == 1:
                    func_str = str(util.simplify(agg_func))
                else:
                    func_str = "(" + ", ".join(str(agg) for agg in agg_func) + ")"
                aggregates.append(f"{group_str}: {func_str}")
            agg_str = ", ".join(agg for agg in aggregates)
            prefix = f"{_make_sub(columns_str)}  " if columns_str else ""
            suffix = f" {_make_sub(agg_str)}" if agg_str else ""
            node_str = "".join([prefix, _make_label("γ"), suffix])
        case relalg.Map():
            pretty_mapping: dict[str, str] = {}
            for target_col, expression in node.mapping.items():
                if len(target_col) == 1:
                    target_col = util.simplify(target_col)
                    target_str = str(target_col)
                else:
                    target_str = "(" + ", ".join(str(t) for t in target_col) + ")"
                if len(expression) == 1:
                    expression = util.simplify(expression)
                    expr_str = str(expression)
                else:
                    expr_str = "(" + ", ".join(str(e) for e in expression) + ")"
                pretty_mapping[target_str] = expr_str
            mapping_str = ", ".join(f"{target_col}: {expr}" for target_col, expr in pretty_mapping.items())
            node_str = f"{_make_label('χ')} {_make_sub(mapping_str)}"
        case _:
            node_str = _escape_label(str(node))
    return f"<{node_str}>", node_params


def _relalg_child_traversal(node: relalg.RelNode) -> Sequence[relalg.RelNode]:
    return node.children()


def plot_relalg(relnode: relalg.RelNode, **kwargs) -> gv.Graph:
    return trees.plot_tree(relnode, _relalg_node_labels, _relalg_child_traversal,
                           escape_labels=False, node_id_generator=id,
                           strict=True,  # gv.Graph arguments
                           **kwargs)
