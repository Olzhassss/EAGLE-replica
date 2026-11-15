"""Resolve logical agent slots and forwarding state from a versioned scenario."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class Scenario:
    fingerprint: str
    controllers: dict
    switches: dict
    hosts: dict
    graph: nx.Graph
    ports: dict
    telemetry: dict


def _by_id(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        id_ = row["id"]
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", id_) or id_ in indexed:
            raise ValueError(f"Invalid or duplicate {label} ID: {id_!r}")
        indexed[id_] = row
    return indexed


def load_scenario(path: str | Path) -> Scenario:
    """Load an undirected switch/host graph and its endpoint port index."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("Expected scenario schema version 1")
    controllers = _by_id(data["controllers"], "controller")
    switches = _by_id(data["switches"], "switch")
    hosts = _by_id(data["hosts"], "host")
    if not controllers or not switches or hosts.keys() & switches.keys():
        raise ValueError(
            "The scenario needs controllers and disjoint switch and host IDs"
        )
    dpids = set()
    for switch in switches.values():
        dpid = switch["dpid"]
        if type(dpid) is not int or not 0 < dpid < (1 << 64) or dpid in dpids:
            raise ValueError(f"Invalid or duplicate datapath ID: {dpid!r}")
        dpids.add(dpid)
        if switch["controller"] not in controllers:
            raise ValueError("Unknown controller for " + switch["id"])

    restrictions = set()
    for controller_id, item in controllers.items():
        support = item["support_switch"]
        if support not in switches or switches[support]["controller"] != controller_id:
            raise ValueError("Supporting switch must belong to " + controller_id)
        if type(item["initial_action"]) is not int or item["initial_action"] not in (
            0,
            1,
        ):
            raise ValueError("Expected a binary initial action for " + controller_id)
        for key in ("http_port", "openflow_port"):
            port = item[key]
            address = (item["host"], port)
            if type(port) is not int or not 0 < port < 65536 or address in restrictions:
                raise ValueError(f"Invalid or duplicate endpoint {address!r}")
            restrictions.add(address)

    graph = nx.Graph()
    for name in switches:
        graph.add_node(name, kind="switch")
    addresses = set()
    for name, host in hosts.items():
        mac = host["mac"].lower()
        ipaddress.ip_interface(host["ip"])
        if mac in addresses or not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", mac):
            raise ValueError("Invalid or duplicate MAC: " + mac)
        addresses.add(mac)
        graph.add_node(name, kind="host", mac=mac)
    ports = {name: {} for name in graph}
    for link in data["links"]:
        source, target = link["source"], link["target"]
        capacity = link["capacity_bps"]
        if source not in graph or target not in graph or source == target:
            raise ValueError(f"Invalid link endpoints: {source} to {target}")
        if graph.has_edge(source, target) or type(capacity) is not int or capacity <= 0:
            raise ValueError(
                f"Duplicate link or invalid link capacity: {source} to {target}"
            )
        endpoints = {source: link["source_port"], target: link["target_port"]}
        for node, neighbor in ((source, target), (target, source)):
            port = endpoints[node]
            minimum = 1 if node in switches else 0
            if type(port) is not int or port < minimum or port in ports[node]:
                raise ValueError(f"Invalid or duplicate {node} port {port}")
            ports[node][port] = neighbor
        graph.add_edge(source, target, ports=endpoints, capacity_bps=capacity)
    for host in hosts:
        if graph.degree(host) != 1 or next(iter(graph[host])) not in switches:
            raise ValueError("Expected one switch attachment for " + host)

    telemetry = data["telemetry"]
    if any(
        telemetry[key] <= 0
        for key in ("poll_seconds", "peer_poll_seconds", "peer_timeout_seconds")
    ):
        raise ValueError("Telemetry intervals must be positive")
    return Scenario(
        hashlib.sha256(raw).hexdigest(),
        controllers,
        switches,
        hosts,
        graph,
        ports,
        telemetry,
    )


class TopologyState:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.configured = scenario.graph
        self.resources = {
            slot: {"action": controller["initial_action"], "revision": [0, ""]}
            for slot, controller in scenario.controllers.items()
        }
        self.active = nx.Graph()
        self.owners = {}
        self.connected = set()

    def merge_resources(self, presence: dict[str, dict[str, Any]]) -> None:
        """Accept the latest acknowledged action for each logical controller slot."""
        for slot, update in presence.items():
            current = self.resources.get(slot)
            if (
                current is not None
                and update["action"] in (0, 1)
                and update["revision"] > current["revision"]
            ):
                self.resources[slot] = update

    def set_support_action(
        self,
        slot: str,
        action: int,
        revision: list[int | str],
    ) -> dict[str, Any]:
        current = self.resources[slot]
        if current["action"] != action:
            self.resources[slot] = {"action": action, "revision": revision}
        return self.resources[slot]

    def update_presence(self, frames: dict[str, dict[str, Any]]) -> None:
        """Elect physical controllers and rebuild the forwarding graph.

        All switches connect to each configured controller. A live primary keeps
        its logical slot. A failed primary's slot moves to the first live peer
        in scenario order. A switch is reachable through its elected owner.
        """
        live = [slot for slot in self.scenario.controllers if slot in frames]
        self.owners = (
            {slot: slot if slot in frames else live[0] for slot in self.resources}
            if live
            else {}
        )
        self.connected = set()
        for switch, row in self.scenario.switches.items():
            frame = frames.get(self.owners.get(row["controller"]), {})
            if switch in frame.get("ports", {}):
                self.connected.add(switch)
        disabled = {
            self.scenario.controllers[slot]["support_switch"]
            for slot, update in self.resources.items()
            if not update["action"]
        }
        active_switches = self.connected - disabled
        self.active = self.configured.subgraph(
            active_switches | self.scenario.hosts.keys()
        ).copy()
        for source, target, edge in list(self.active.edges(data=True)):
            endpoints = ((source, target), (target, source))
            for node, neighbor in endpoints:
                if node not in active_switches:
                    continue
                slot = self.scenario.switches[node]["controller"]
                active_ports = frames[self.owners[slot]]["ports"][node]
                if edge["ports"][node] not in active_ports:
                    self.active.remove_edge(node, neighbor)
                    break
        self.active.remove_nodes_from(
            host for host in self.scenario.hosts if not self.active.degree(host)
        )

    def observation_graph(self, slot: str, view: str) -> nx.Graph:
        """Include the subnet and links incident on its switches."""
        graph = self.active if view == "active" else self.configured
        local = {
            node
            for node, row in self.scenario.switches.items()
            if row["controller"] == slot
        }
        local &= graph.nodes.keys()
        observed = nx.Graph()
        observed.add_nodes_from((node, graph.nodes[node]) for node in sorted(local))
        for source, target, edge in graph.edges(data=True):
            if source in local or target in local:
                observed.add_node(source, **graph.nodes[source])
                observed.add_node(target, **graph.nodes[target])
                observed.add_edge(source, target, **edge)
        return observed

    def resource_usage(self, slot: str) -> tuple[int, int]:
        """Count active local switches and ports for modeled power."""
        switches = [
            node
            for node in self.active
            if node in self.scenario.switches
            and self.scenario.switches[node]["controller"] == slot
        ]
        return len(switches), sum(self.active.degree(node) for node in switches)
