"""Partition weighted, undirected switch graphs subject to controller capacity."""

import math
import random
from collections import defaultdict
from collections.abc import Iterator

import networkx as nx


def balanced_partitions(
    graph: nx.Graph,
    *,
    capacity: int,
    difference: int,
    rho: float,
    resolution: float,
    seed: int,
    weight: str = "weight",
) -> Iterator[list[set[int | str]]]:
    """Yield disjoint sets of original node IDs at each aggregation level.

    A move must leave the destination within capacity. Subtract rho from its
    modularity gain if the largest and smallest remaining communities differ
    by more than difference. Empty communities do not count toward balance.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Normalize to a simple undirected graph before partitioning")
    if capacity < 1 or difference < 0 or rho < 0 or resolution <= 0:
        raise ValueError(
            "Capacity and resolution must be positive, difference and rho nonnegative"
        )
    if not math.isfinite(rho) or not math.isfinite(resolution):
        raise ValueError("rho and resolution must be finite")
    network = nx.Graph()
    network.add_nodes_from((node, {"members": {node}}) for node in graph)
    for source, target, attrs in graph.edges(data=True):
        value = float(attrs.get(weight, 1))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Louvain weights must be finite and nonnegative")
        network.add_edge(source, target, weight=value)
    rng = random.Random(seed)
    if not network.size(weight="weight"):
        yield [{node} for node in network]
        return
    while True:
        communities = _move_nodes(network, capacity, difference, rho, resolution, rng)
        yield [
            set().union(*(network.nodes[node]["members"] for node in group))
            for group in communities
        ]
        if len(communities) == len(network) or len(communities) < 2:
            return
        network = _contract(network, communities)


def partition_domains(
    graph: nx.Graph,
    domains: int,
    *,
    capacity: int,
    difference: int,
    rho: float,
    resolution: float,
    seed: int,
    weight: str,
) -> tuple[list[set[int]], list[set[int]]]:
    """Return the connected groups before merging and the final domains."""
    groups = [{node} for node in graph]
    for level in balanced_partitions(
        graph,
        capacity=capacity,
        difference=difference,
        rho=rho,
        resolution=resolution,
        seed=seed,
        weight=weight,
    ):
        if len(level) < domains:
            break
        groups = level
    groups = [
        component
        for group in groups
        for component in nx.connected_components(graph.subgraph(group))
    ]
    if len(groups) == domains:
        return groups, groups

    # Merge whole connected groups on the quotient to reach the requested count.
    quotient = _contract(graph, groups, weight)
    merged = nx.community.greedy_modularity_communities(
        quotient,
        weight="weight",
        resolution=resolution,
        cutoff=domains,
        best_n=domains,
    )
    return groups, [set().union(*(groups[index] for index in group)) for group in merged]


def _move_nodes(
    graph: nx.Graph,
    capacity: int,
    difference: int,
    rho: float,
    resolution: float,
    rng: random.Random,
) -> list[set[str | int]]:
    order = list(graph)
    rng.shuffle(order)
    names = {node: index for index, node in enumerate(graph)}
    groups = [{node} for node in graph]
    weights = dict(graph.degree(weight="weight"))
    totals = [weights[node] for node in graph]
    sizes = [len(graph.nodes[node]["members"]) for node in graph]
    mass = graph.size(weight="weight")

    while True:
        moved = False
        for node in order:
            origin = names[node]
            degree = weights[node]
            size = len(graph.nodes[node]["members"])
            adjacent = defaultdict(float)
            for other, edge in graph[node].items():
                if other != node:
                    adjacent[names[other]] += edge["weight"]
            totals[origin] -= degree
            penalty = resolution * degree / (2 * mass * mass)
            origin_score = adjacent[origin] / mass - penalty * totals[origin]
            best, best_gain = origin, 1e-12
            for candidate, connection in sorted(adjacent.items()):
                if candidate == origin or sizes[candidate] + size > capacity:
                    continue
                gain = connection / mass - penalty * totals[candidate] - origin_score
                if gain <= best_gain:
                    continue
                new_sizes = sizes.copy()
                new_sizes[origin] -= size
                new_sizes[candidate] += size
                active_sizes = [count for count in new_sizes if count]
                if max(active_sizes) - min(active_sizes) > difference:
                    gain -= rho
                if gain > best_gain:
                    best, best_gain = candidate, gain
            totals[best] += degree
            if best != origin:
                sizes[origin] -= size
                sizes[best] += size
                groups[origin].remove(node)
                groups[best].add(node)
                names[node] = best
                moved = True
        if not moved:
            return [group for group in groups if group]


def _contract(
    graph: nx.Graph, groups: list[set[int | str]], weight: str = "weight"
) -> nx.Graph:
    parent = {node: index for index, group in enumerate(groups) for node in group}
    coarse = nx.Graph()
    for index, group in enumerate(groups):
        coarse.add_node(
            index,
            members=set().union(
                *(graph.nodes[node].get("members", {node}) for node in group)
            ),
        )
    for source, target, edge in graph.edges(data=True):
        a, b = parent[source], parent[target]
        accumulated = coarse.get_edge_data(a, b, {}).get("weight", 0)
        coarse.add_edge(a, b, weight=accumulated + edge[weight])
    return coarse
