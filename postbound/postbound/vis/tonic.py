
from __future__ import annotations

from typing import Optional

import graphviz as gv

from postbound.optimizer.strategies import tonic

def _render_subquery_path(qeps: tonic.QepsNode, current_node: str, current_graph: gv.Digraph) -> None:
    for identifier, qeps_child in qeps.child_nodes.items():
        child_node = _make_node_label(identifier, qeps_child)
        current_graph.node(child_node, style="dashed")
        current_graph.edge(current_node, child_node, style="dashed")
        _render_subquery_path(qeps_child, child_node, current_graph)


def _make_node_label(identifier: tonic.QepsIdentifier, node: tonic.QepsNode) -> str:
    cost_str = ("[" + ", ".join(f"{operator.value}={cost}" for operator, cost in node.operator_costs.items()) + "]"
                if node.operator_costs else "")
    label = str(identifier)
    return label + "\n" + cost_str


def plot_tonic_qeps(qeps: tonic.QepsNode, *, _current_node: Optional[str],
                    _current_graph: Optional[gv.Digraph] = None) -> gv.Digraph:
    if not _current_graph:
        _current_graph = gv.Digraph()

    if qeps.subquery_root:
        _render_subquery_path(qeps.subquery_root, _current_node, _current_graph)

    for identifier, qeps_child in qeps.child_nodes.items():
        child_node = _make_node_label(identifier, qeps_child)
        if qeps_child.subquery_root:
            _current_graph.node(child_node, style="dashed")
        else:
            _current_graph.node(child_node)
        if _current_node:
            _current_graph.edge(_current_node, child_node)
        plot_tonic_qeps(qeps_child, _current_node=child_node, _current_graph=_current_graph)

    return _current_graph
