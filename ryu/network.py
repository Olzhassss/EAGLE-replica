"""Read the generated network document into the Ryu graph and port index."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import networkx as nx

from network_schema import validate_support_layout


@dataclass
class Network:
    """Configured controller slots, node addresses, links, and telemetry settings."""

    fingerprint: str
    controllers: dict
    switches: dict
    hosts: dict
    graph: nx.Graph
    ports: dict
    telemetry: dict


def load_network(path: Union[str, Path]) -> Network:
    """Build the switch/host graph and index each local port by its neighbor."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    validate_support_layout(data)
    controllers = {row["id"]: row for row in data["controllers"]}
    switches = {row["id"]: row for row in data["switches"]}
    hosts = {row["id"]: row for row in data["hosts"]}
    graph = nx.Graph()
    graph.add_nodes_from((name, {"kind": "switch"}) for name in switches)
    graph.add_nodes_from(
        (name, {"kind": "host", "mac": host["mac"].lower()})
        for name, host in hosts.items()
    )
    ports = {name: {} for name in graph}
    for link in data["links"]:
        source, target = link["source"], link["target"]
        endpoints = {source: link["source_port"], target: link["target_port"]}
        ports[source][endpoints[source]] = target
        ports[target][endpoints[target]] = source
        graph.add_edge(source, target, ports=endpoints, capacity_bps=link["capacity_bps"])
    return Network(
        hashlib.sha256(raw).hexdigest(),
        controllers,
        switches,
        hosts,
        graph,
        ports,
        data["telemetry"],
    )
