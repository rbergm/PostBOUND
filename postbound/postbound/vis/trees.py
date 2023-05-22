from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv

T = typing.TypeVar("T")


def _gv_escape(content: str) -> str:
    return "".join(c for c in content if c.isalnum())


def plot_tree(node: T, label_generator: Callable[[T], tuple[str, dict]], child_supplier: Callable[[T], Sequence[T]], *,
              __graph: Optional[gv.Graph] = None) -> gv.Graph:
    __graph = gv.Graph() if __graph is None else __graph
    label, params = label_generator(node)
    node_key = _gv_escape(str(node))
    __graph.node(gv.escape(node_key), label=label, **params)

    for child in child_supplier(node):
        child_key = _gv_escape(str(child))
        __graph.edge(node_key, child_key)
        __graph = plot_tree(child, label_generator, child_supplier, __graph=__graph)
    return __graph
