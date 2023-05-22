"""Utilities to visualize different aspects of query optimization, namely query plans."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv

from postbound.db import db
from postbound.vis import trees as tree_viz


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


def _query_plan_traversal(node: db.QueryExecutionPlan) -> Sequence[db.QueryExecutionPlan]:
    if node.is_scan:
        return ()
    elif node.is_join and node.inner_child:
        return (node.outer_child, node.inner_child)
    return node.children


def plot_query_plan(plan: db.QueryExecutionPlan,
                    annotation_generator: Optional[Callable[[db.QueryExecutionPlan], str]] = None) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return tree_viz.plot_tree(plan, functools.partial(_query_plan_labels, annotation_generator=annotation_generator),
                              _query_plan_traversal)
