from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Optional

import graphviz as gv

T = typing.TypeVar("T")


def plot_tree(node: T, label_generator: callable[[T], str], child_supplier: callable[[T], Sequence[T]], *,
              __graph: Optional[gv.Graph] = None) -> gv.Graph:
    __graph = gv.Graph() if __graph is None else __graph
    __graph.node(str(node), label=label_generator(node))

    for child in child_supplier(node):
        __graph.edge(str(node), str(child))
        __graph = plot_tree(child, label_generator, child_supplier, __graph=__graph)
    return __graph
