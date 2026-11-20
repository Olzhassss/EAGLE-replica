"""Pure planning for Ethernet flows in the active network graph."""

from typing import Dict, Tuple

import crc32c
import networkx as nx

from ryu.network import Network

ForwardingRules = Dict[Tuple[str, str], int]


def plan_forwarding_rules(
    graph: nx.Graph,
    network: Network,
) -> Dict[str, ForwardingRules]:
    """Map each active switch to {(source MAC, destination MAC): output port}.

    Pick one end-to-end shortest path for each ordered host pair. Prefer paths
    through a source-domain support switch and hash across equal choices.
    """
    hosts = sorted(node for node in network.hosts if node in graph)
    rules = {node: {} for node in network.switches}
    for source in hosts:
        attachment = next(iter(graph[source]))
        slot = network.switches[attachment]["controller"]
        support = set(network.controllers[slot]["support_switches"])
        for target in hosts:
            if source == target:
                continue
            try:
                candidates = sorted(nx.all_shortest_paths(graph, source, target))
            except nx.NetworkXNoPath:
                continue
            preferred = [path for path in candidates if support.intersection(path)]
            paths = preferred or candidates
            pair = (graph.nodes[source]["mac"], graph.nodes[target]["mac"])
            key = (source + target + "".join(pair)).encode()
            path = paths[crc32c.crc32c(key) % len(paths)]
            for switch, next_hop in zip(path[1:-1], path[2:]):  # noqa: B905, RUF007
                rules[switch][pair] = graph[switch][next_hop]["ports"][switch]
    return rules
