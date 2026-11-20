"""Choose a minimum-hop controller location inside each community."""

from __future__ import annotations
from collections.abc import Iterable
from itertools import islice, zip_longest
from typing import Any

import networkx as nx

from network_schema import resolve_support_counts


def place_controllers(
    graph: nx.Graph,
    communities: Iterable[set[int | str]],
) -> list[dict[str, Any]]:
    """Return deterministic unit-load assignments and controller-to-switch distances."""
    placements = []
    for community in communities:
        members = sorted(community)
        if not members:
            continue
        options = []
        for node in members:
            lengths = nx.single_source_shortest_path_length(graph, node)
            if all(member in lengths for member in members):
                options.append((sum(lengths[member] for member in members), node))
        if not options:
            raise ValueError("Community has no reachable controller placement")
        hops, node = min(options)
        placements.append(
            {
                "switches": members,
                "placement_switch": node,
                "mean_hops": hops / len(members),
            }
        )
    return sorted(placements, key=lambda row: row["placement_switch"])


def select_controllable(
    graph: nx.Graph,
    domains: list[set[int | str]],
    counts: int | Iterable[int],
    *,
    excluded: Iterable[int | str] = (),
) -> list[list[int | str]]:
    """Select support switches while keeping every disable combination connected."""
    counts = resolve_support_counts(counts, len(domains))
    owner = {node: index for index, group in enumerate(domains) for node in group}
    if sum(counts) >= len(graph):
        raise ValueError("Controllable switches must leave at least one active switch")
    ranked = sorted(
        set(graph) - set(nx.articulation_points(graph)) - set(excluded),
        key=lambda node: (-graph.degree(node), node),
    )
    chosen = [[] for _ in domains]
    pending = list(counts)

    def select(start: int, disabled: set[int | str]) -> bool:
        """Backtrack over candidates while preserving connectivity for every disabled subset."""
        if not any(pending):
            return True
        active = graph.subgraph(graph.nodes - disabled)
        cuts = set(nx.articulation_points(active))
        candidates = [
            (index, node)
            for index, node in enumerate(ranked[start:], start)
            if pending[owner[node]] and node not in cuts
        ]
        available = [0] * len(domains)
        for _, node in candidates:
            available[owner[node]] += 1
        if any(need > count for need, count in zip(pending, available, strict=True)):
            return False

        for index, node in candidates:
            removed = disabled | {node}
            # Any selected switch may be the only one re-enabled later.
            if any(not (graph[node].keys() - removed) for node in removed):
                continue
            group = owner[node]
            chosen[group].append(node)
            pending[group] -= 1
            if select(index + 1, removed):
                return True
            pending[group] += 1
            chosen[group].pop()
        return False

    if not select(0, set()):
        raise ValueError(
            f"Cannot select support-switch counts {counts} while preserving connectivity"
        )
    return chosen


def assign_hosts(
    graph: nx.Graph, domains: list[set[int]], controllable: list[list[int]], hosts: int
) -> list[int]:
    """Fill host slots round-robin across domains, starting at the periphery."""
    eligible = [
        sorted(group.difference(reserved), key=lambda node: (graph.degree(node), node))
        for group, reserved in zip(domains, controllable, strict=True)
    ]
    capacity = sum(map(len, eligible))
    if hosts > capacity:
        raise ValueError(
            f"{hosts} hosts requested but only {capacity} access switches are available"
        )
    return list(
        islice(
            (
                node
                for batch in zip_longest(*eligible)
                for node in batch
                if node is not None
            ),
            hosts,
        )
    )
