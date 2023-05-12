"""Force-directed layout algorithms"""
from __future__ import annotations

import collections
import typing
from collections.abc import Iterable

import networkx as nx
import numpy as np

T = typing.TypeVar("T")


def force_directed_layout(elements: Iterable[T], difference_score: callable[[T, T], float]) -> dict[T, np.ndarray]:
    """Lays out the supplied elements in a 2D-space according to the difference score.

    Pairs of points with a large difference score are positioned further apart than points with a low difference score.

    The returned dictionary maps each of the input element to the pair of (x, y) coordinates.
    """
    elements = list(elements)
    layout_graph = nx.complete_graph(elements)

    distance_map = collections.defaultdict(dict)
    for a_idx, a in enumerate(elements):
        for b in elements[a_idx:]:
            current_score = difference_score(a, b)
            distance_map[a][b] = current_score
            distance_map[b][a] = current_score

    return nx.kamada_kawai_layout(layout_graph, dist=distance_map)
