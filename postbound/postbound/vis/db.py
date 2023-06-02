"""Utilities to visualize different aspects of query optimization, namely query plans."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv

from postbound.db import db
from postbound.vis import trees as tree_viz
from postbound.util import collections as collection_utils


def _query_plan_labels(node: db.QueryExecutionPlan, *,
                       annotation_generator: Optional[Callable[[db.QueryExecutionPlan], str]]) -> tuple[str, dict]:
    if node.is_join:
        label, params = node.node_type, {"style": "bold"}
    elif node.is_scan:
        label, params = f"<<{node.node_type}>>\n{node.table}", {"color": "grey"}
    else:
        label, params = node.node_type, {"style": "dashed", "color": "grey"}

    annotation = annotation_generator(node) if annotation_generator else ""
    label = f"{label}\n{annotation}" if annotation else label
    return label, params


def _query_plan_traversal(node: db.QueryExecutionPlan, *,
                          skip_intermediates: bool = False) -> Sequence[db.QueryExecutionPlan]:
    if not node.children:
        return ()

    if node.is_scan:
        return ()
    elif node.is_join and node.inner_child:
        children = node.outer_child, node.inner_child
    else:
        children = node.children

    if skip_intermediates:
        skipped = [_query_plan_traversal(child, skip_intermediates=True) if not child.is_scan and not child.is_join
                   else child for child in children]
        children = collection_utils.flatten(skipped)
    return children


def annotate_estimates(node: db.QueryExecutionPlan) -> str:
    return f"cost={node.cost} cardinality={node.estimated_cardinality}"


def plot_query_plan(plan: db.QueryExecutionPlan,
                    annotation_generator: Optional[Callable[[db.QueryExecutionPlan], str]] = None, *,
                    skip_intermediates: bool = False) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return tree_viz.plot_tree(plan, functools.partial(_query_plan_labels, annotation_generator=annotation_generator),
                              functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates))


def _explain_analyze_annotations(node: db.QueryExecutionPlan) -> str:
    card_row = f"[Rows expected={node.estimated_cardinality} actual={node.true_cardinality}]"
    exec_time = round(node.execution_time, 4)
    runtime_row = f"[Exec time={exec_time}s]"
    return card_row + "\n" + runtime_row


def plot_analyze_plan(plan: db.QueryExecutionPlan, *, skip_intermediates: bool = False) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return tree_viz.plot_tree(plan,
                              functools.partial(_query_plan_labels, annotation_generator=_explain_analyze_annotations),
                              functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates))
