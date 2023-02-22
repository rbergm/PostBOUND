"""Contains additional algorithms tailored to PostBOUND to work with networkx graphs."""
from __future__ import annotations

import random
import typing
from typing import Callable

import networkx as nx

from postbound.util import collections as collection_utils

NodeType = typing.TypeVar("NodeType")
EdgeType = typing.TypeVar("EdgeType")


def nx_random_walk(graph: nx.Graph) -> EdgeType:
    """A modified random walk implementation for networkx graphs.

    The modifications concern two specific areas: after each stop, the walk may jump to a node that is connected to one of the
    visited nodes. This node does not necessarily have to be connected to the current node. Secondly, if the graph contains
    multiple connected components, the walk will first explore one component before jumping to the next one.
    """
    shell_nodes = set()
    visited_nodes = set()

    total_n_nodes = len(graph.nodes)

    current_node = random.choice(list(graph.nodes))
    visited_nodes.add(current_node)
    yield current_node

    while len(visited_nodes) < total_n_nodes:
        shell_nodes |= set(n for n in graph.adj[current_node].keys() if n not in visited_nodes)
        if not shell_nodes:
            # we have multiple connected components and need to jump into the other component
            current_node = random.choice([n for n in graph.nodes if n not in visited_nodes])
            visited_nodes.add(current_node)
            yield current_node
            continue

        current_node = random.choice(list(shell_nodes))
        shell_nodes.remove(current_node)
        visited_nodes.add(current_node)
        yield current_node


def nx_bfs_tree(graph: nx.Graph, start_node: NodeType,
                condition: Callable[[NodeType, dict], bool], *,
                node_order: Callable[[NodeType, dict], int] | None = None) -> tuple[NodeType, dict]:
    """Traverses the given `graph` in breadth-first manner, beginning at `start_node`.

    This function will yield all encountered nodes if they satisfy the `condition`. If no more nodes are found or the
    condition cannot be satisfied for any more nodes, traversal terminates. Each condition check receives the
    neighboring node along with the edge-data as arguments.
    """
    shell_nodes = collection_utils.Queue([(node, edge) for node, edge in graph.adj[start_node].items()])
    visited_nodes = {start_node}
    while shell_nodes:
        current_node, current_edge = shell_nodes.pop()
        visited_nodes.add(current_node)
        if condition(current_node, current_edge):
            neighbor_nodes = [(node, edge) for node, edge in graph.adj[current_node].items()
                              if node not in visited_nodes]
            if node_order:
                sorted(neighbor_nodes, key=lambda neighbor: node_order(neighbor[0], neighbor[1]))
            shell_nodes.extend(neighbor_nodes)
            yield current_node, current_edge
