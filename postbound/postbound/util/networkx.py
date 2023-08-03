"""Contains additional algorithms tailored to PostBOUND to work with networkx graphs."""
from __future__ import annotations

import dataclasses
import random
import typing
from collections.abc import Sequence
from typing import Callable, Generator, Iterator, Optional

import networkx as nx

from postbound.util import collections as collection_utils

NodeType = typing.TypeVar("NodeType")


def nx_random_walk(graph: nx.Graph) -> Generator[NodeType]:
    """A modified random walk implementation for networkx graphs.

    The modifications concern two specific areas: after each stop, the walk may jump to a node that is connected to
    one of the visited nodes. This node does not necessarily have to be connected to the current node. Secondly, if the
    graph contains multiple connected components, the walk will first explore one component before jumping to the next
    one.
    """
    # TODO: could be refactored to use the GraphWalk class instead
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
                node_order: Callable[[NodeType, dict], int] | None = None) -> Generator[tuple[NodeType, dict]]:
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


@dataclasses.dataclass
class GraphWalk:
    """A `GraphWalk` models an arbitrary traversal of a graph.

    Each walk begins at a `start_node` and then follows along the `path`. The path itself consists of pairs of target
    nodes along with the edge that was traversed to get to the target node.

    Notice that depending on the specific use-case the `path` might deviate from a normal walk. More specifically, two
    special cases might occur:

    1. the edge data can be `None`. This indicates that the walk jumped to a different node without using an edge. For
    example, this might happen if the walk moved to a different connected component of the graph.
    2. the next node in the walk might not be connected to the current node in the path, but to some node that has
    already been visited instead. This is especially the case for so-called _frontier walks_ which can be computed
    using the `nx_frontier_walks` method.
    """
    start_node: NodeType
    path: Sequence[tuple[NodeType, dict]] = dataclasses.field(default_factory=list)

    def nodes(self) -> Sequence[NodeType]:
        """Provides all nodes that are visited by this walk, in the sequence in which they are visited."""
        return [self.start_node] + [node[0] for node in self.path]

    def final_node(self) -> NodeType:
        """Provides the very last node that was visited by this walk."""
        return self.start_node if not self.path else self.path[-1][0]

    def expand(self, next_node: NodeType, edge_data: Optional[dict] = None) -> GraphWalk:
        """Creates a new walk by prolonging the current one with one more edge at the end."""
        return GraphWalk(self.start_node, list(self.path) + [(next_node, edge_data)])

    def nodes_hash(self) -> int:
        """Provides a hash value only based on the nodes sequence, not the selected predicates."""
        return hash(tuple(self.nodes()))

    def __len__(self) -> int:
        return 1 + len(self.path)

    def __iter__(self) -> Iterator[NodeType]:
        return self.nodes().__iter__()

    def __contains__(self, other: object) -> bool:
        return other in self.nodes()

    def __hash__(self) -> int:
        return hash((self.start_node, tuple(self.path)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.start_node == other.start_node and self.path == other.path

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " -> ".join(str(node) for node in self.nodes())


def _walk_frontier(graph: nx.Graph, current_walk: GraphWalk, current_frontier: set[NodeType]) -> Generator[GraphWalk]:
    """Expands the current walk based on the nodes available in the frontier in a recusive manner."""
    available_edges = []
    for frontier_node in current_frontier:
        current_edges = graph.adj[frontier_node]
        current_edges = [(target_node, edge_data) for target_node, edge_data in current_edges.items()
                         if target_node not in current_walk and target_node not in current_frontier]
        available_edges.extend(current_edges)

    if not available_edges and len(current_walk) < len(graph):
        jump_nodes = [node for node in graph.nodes if node not in current_frontier]
        for jump_node in jump_nodes:
            yield from _walk_frontier(graph, current_walk.expand(jump_node), current_frontier | {jump_node})
    elif not available_edges:
        yield current_walk
    else:
        for target_node, edge_data in available_edges:
            yield from _walk_frontier(graph, current_walk.expand(target_node, edge_data),
                                      current_frontier | {target_node})


def nx_frontier_walks(graph: nx.Graph) -> Generator[GraphWalk]:
    """Provides all possible frontier walks over the given graph.

    A _frontier walk_ is a generalized version of a normal walk over a graph: Whereas a normal walk traverses the
    edges in the graph to move from node to node in a local fashion (i.e. only based on the edges of the current node),
    a fronteir walk remembers all the nodes that have already been visited. This is called the _frontier_ of the
    current walk. To find the next node, any edge from any of the nodes in the frontier can be selected.

    Notice that the frontier walk also remembers nodes that have already been visited and prevents them from being
    visited again.

    Our implementation augments this procedure by also allowing jumps to other partitions in the graph. This will
    happen if all nodes in the current connected component have been visited, but more unexplored nodes remain.

    Notice that this method already distinguishes between paths if they differ in the traversed edges, even if the
    sequence of nodes is the same.

    For example, consider this fully-connected graph:
    ```
      a
     / \\
    b - c
    ```
    The frontier walks produced by this function will include the sequence `a -> b -> c` twice (among many other
    sequences):
    Once by traversing the edge `a -> c` to reach `c`, and once by traversing the edge `b -> c` to reach `c`.
    """
    for node in graph.nodes:
        yield from _walk_frontier(graph, GraphWalk(node), current_frontier={node})
