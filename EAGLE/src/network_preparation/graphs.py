"""Generate random switch graphs and normalize GraphML topology inputs."""

from itertools import combinations
from pathlib import Path
from random import Random

import networkx as nx


def random_switch_graph(nodes: int, links: int, seed: int) -> nx.Graph:
    """Create a connected simple graph with exactly the requested sizes."""
    if nodes < 1 or not nodes - 1 <= links <= nodes * (nodes - 1) // 2:
        raise ValueError(
            "A random graph needs positive nodes and between n−1 and n(n−1)/2 links"
        )
    rng = Random(seed)
    graph = nx.Graph()
    switches = list(range(1, nodes + 1))
    graph.add_nodes_from(switches)
    order = rng.sample(switches, nodes)
    for index, switch in enumerate(order[1:], start=1):
        graph.add_edge(switch, order[rng.randrange(index)])
    extra = links - graph.number_of_edges()
    if extra:
        available = [
            (a, b) for a, b in combinations(switches, 2) if not graph.has_edge(a, b)
        ]
        graph.add_edges_from(rng.sample(available, extra))
    return graph


def load_switch_graph(path: Path) -> nx.Graph:
    """Project a Zoo graph to the largest connected simple switch component."""
    original = nx.read_graphml(path)
    switches = {
        node
        for node, data in original.nodes(data=True)
        if data.get("kind") != "host" and str(data.get("isHost")).lower() != "true"
    }
    graph = nx.Graph()
    graph.add_nodes_from(sorted(switches))
    graph.add_edges_from(
        sorted(
            {
                tuple(sorted((a, b)))
                for a, b in original.edges()
                if a != b and a in switches and b in switches
            }
        )
    )
    if not graph:
        raise ValueError(f"{path.name} has no switches")
    component = max(
        nx.connected_components(graph), key=lambda nodes: (len(nodes), sorted(nodes))
    )
    ids = {node: index for index, node in enumerate(sorted(component), start=1)}
    result = nx.Graph()
    result.add_nodes_from((index, {"source_node": node}) for node, index in ids.items())
    result.add_edges_from(
        (ids[a], ids[b]) for a, b in graph.edges if a in ids and b in ids
    )
    return result
