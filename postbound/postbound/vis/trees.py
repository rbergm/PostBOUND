"""Provides generic utilities to transform arbitrary graph-like structures into Graphviz objects."""
from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from typing import Optional

import graphviz as gv

T = typing.TypeVar("T")
"""Node type of the tree to plot."""


def _gv_escape(node: T) -> str:
    """Generates an (hopefully unique) identifier of a specific node.

    The identifier is based on the nodes name as well as its hash value.

    Parameters
    ----------
    node : T
        The node to generate the identifier for.

    Returns
    -------
    str
        The identifier.

    Warnings
    --------
    If two nodes have the same textual representation (according to their *str* methods), as well as the same hash value, they
    will be recognized as the same logical nodes.
    """
    content = str(node) + str(hash(node))
    return "".join(c for c in content if c.isalnum())


def plot_tree(node: T, label_generator: Callable[[T], tuple[str, dict]], child_supplier: Callable[[T], Sequence[T]], *,
              escape_labels: bool = True, out_path: str = "", out_format: str = "svg",
              _graph: Optional[gv.Graph] = None, **kwargs) -> gv.Graph:
    """Transforms an arbitrary tree into a Graphviz graph. The tree traversal is achieved via callback functions.

    Start the traversal at the root node.

    Parameters
    ----------
    node : T
        The node to plot.
    label_generator : Callable[[T], tuple[str, dict]]
        Callback function to generate labels of the nodes in the graph. The dictionary can contain additional formatting
        attributes (e.g. bold font). Consult the Graphviz documentation for allowed values
    child_supplier : Callable[[T], Sequence[T]]
        Provides the children of the current node.
    out_path : str, optional
        An optional file path to store the graph at. If empty, the graph will only be provided as a Graphviz object.
    out_format : str, optional
        The output format of the graph. Defaults to SVG and will only be used if the graph should be stored to disk (according
        to `out_path`).
    _graph : Optional[gv.Graph], optional
        Internal parameter used for state-management within the plotting function. Do not set this parameter yourself!

    Returns
    -------
    gv.Graph
        _description_

    See Also
    --------
    gv.Dot.node
    gv.Dot.edge

    References
    ----------

    .. Graphviz project: https://graphviz.org/
    """
    initial = _graph is None
    _graph = gv.Graph(**kwargs) if initial else _graph
    label, params = label_generator(node)
    if escape_labels:
        label = gv.escape(label)
    node_key = _gv_escape(node)
    _graph.node(node_key, label=label, **params)

    for child in child_supplier(node):
        child_key = _gv_escape(child)
        _graph.edge(node_key, child_key)
        _graph = plot_tree(child, label_generator, child_supplier, escape_labels=escape_labels, _graph=_graph)

    if initial and out_path:
        _graph.render(out_path, format=out_format, cleanup=True)
    return _graph
