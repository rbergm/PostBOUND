"""Contains additional algorithms tailored to PostBOUND to work with networkx graphs."""
import random

import networkx as nx


def nx_random_walk(graph: nx.Graph):
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
