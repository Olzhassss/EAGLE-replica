"""Choose a minimum-hop controller location inside each community."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx


def place_controllers(
    graph: nx.Graph,
    communities: Iterable[set[str]],
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
