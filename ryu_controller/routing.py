"""Pure planning for Ethernet flows in the active scenario graph."""

from __future__ import annotations

import crc32c
import networkx as nx

from .topology_state import Scenario

ForwardingRules = dict[tuple[str, str], int]


def plan_forwarding_rules(
    graph: nx.Graph,
    scenario: Scenario,
) -> dict[str, ForwardingRules]:
    """Map each active switch to {(source MAC, destination MAC): output port}.

    Pick one end-to-end shortest path for each ordered host pair. Prefer paths
    through the source domain's supporting switch and hash across equal choices.
    """
    hosts = sorted(node for node in scenario.hosts if node in graph)
    rules = {node: {} for node in scenario.switches}
    for source in hosts:
        attachment = next(iter(graph[source]))
        slot = scenario.switches[attachment]["controller"]
        support = scenario.controllers[slot]["support_switch"]
        for target in hosts:
            if source == target:
                continue
            try:
                candidates = sorted(nx.all_shortest_paths(graph, source, target))
            except nx.NetworkXNoPath:
                continue
            preferred = [path for path in candidates if support in path]
            paths = preferred or candidates
            pair = (graph.nodes[source]["mac"], graph.nodes[target]["mac"])
            key = (source + target + "".join(pair)).encode()
            path = paths[crc32c.crc32c(key) % len(paths)]
            for switch, next_hop in zip(path[1:-1], path[2:]):
                rules[switch][pair] = graph[switch][next_hop]["ports"][switch]
    return rules
