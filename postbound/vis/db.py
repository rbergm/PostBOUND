"""Utilities to visualize different aspects of query optimization, namely query plans."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv

from . import trees
from .. import util
from ..db import QueryExecutionPlan


def _query_plan_labels(node: QueryExecutionPlan, *,
                       annotation_generator: Optional[Callable[[QueryExecutionPlan], str]],
                       _in_subplan: bool = False) -> tuple[str, dict]:
    if node.is_subplan_root and not _in_subplan:
        label, params = _query_plan_labels(node, annotation_generator=annotation_generator, _in_subplan=True)
        subplan_name = (" " + node.subplan_name) if node.subplan_name else ""
        label = f"<<SubPlan>>{subplan_name}\n{label}"
        params["style"] = "dashed"
    elif node.is_join:
        label, params = node.node_type, {"style": "bold"}
    elif node.is_scan:
        label, params = f"<<{node.node_type}>>\n{node.table}", {"color": "grey"}
    else:
        label, params = node.node_type, {"style": "dashed", "color": "grey"}

    if not node.is_subplan_root and node.subplan_name:
        label = f"{label}\nSubplan: {node.subplan_name}"

    annotation = annotation_generator(node) if annotation_generator else ""
    label = f"{label}\n{annotation}" if annotation else label
    return label, params


def _query_plan_traversal(node: QueryExecutionPlan, *,
                          skip_intermediates: bool = False) -> Sequence[QueryExecutionPlan]:
    children = ()
    if node.subplan_input:
        children = (node.subplan_input,)

    if not node.children:
        return children

    if node.is_scan:
        return children
    elif node.is_join and node.inner_child:
        children = node.outer_child, node.inner_child
    else:
        children = node.children

    if node.subplan_input:
        children = children + (node.subplan_input,)

    if skip_intermediates:
        skipped = [_query_plan_traversal(child, skip_intermediates=True) if not child.is_scan and not child.is_join
                   else child for child in children]
        children = util.flatten(skipped)
    return children


def annotate_estimates(node: QueryExecutionPlan) -> str:
    return f"cost={node.cost} cardinality={node.estimated_cardinality}"


def plot_query_plan(plan: QueryExecutionPlan,
                    annotation_generator: Optional[Callable[[QueryExecutionPlan], str]] = None, *,
                    skip_intermediates: bool = False, **kwargs) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return trees.plot_tree(plan, functools.partial(_query_plan_labels, annotation_generator=annotation_generator),
                           functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates),
                           **kwargs)


def _explain_analyze_annotations(node: QueryExecutionPlan) -> str:
    card_row = f"[Rows expected={node.estimated_cardinality} actual={node.true_cardinality}]"
    exec_time = round(node.execution_time, 4)
    runtime_row = f"[Exec time={exec_time}s]"
    return card_row + "\n" + runtime_row


def plot_analyze_plan(plan: QueryExecutionPlan, *, skip_intermediates: bool = False, **kwargs) -> gv.Graph:
    if not plan:
        return gv.Graph()
    return trees.plot_tree(plan,
                           functools.partial(_query_plan_labels, annotation_generator=_explain_analyze_annotations),
                           functools.partial(_query_plan_traversal, skip_intermediates=skip_intermediates),
                           **kwargs)
