"""Load attributed switch graphs and export controller-assigned scenarios."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import networkx as nx

TOPOLOGY_ZOO = Path(__file__).with_name("topology-zoo")


def load_switch_graph(path: str | Path) -> nx.Graph:
    """Keep switch identities and collapse directed/parallel input links.

    The largest connected component is used for GraphML benchmarks. Scenario
    JSON must be connected and retains exactly its configured switch IDs.
    """
    path = Path(path)
    if path.suffix.lower() == ".graphml":
        source = nx.read_graphml(path)
        simple = nx.Graph()
        simple.add_nodes_from(
            node
            for node, data in source.nodes(data=True)
            if data.get("kind") != "host" and str(data.get("isHost")).lower() != "true"
        )
        for a, b, attrs in source.edges(data=True):
            if a in simple and b in simple:
                value = float(attrs.get("weight", 1))
                existing = simple.get_edge_data(a, b, {}).get("weight", 0)
                simple.add_edge(a, b, weight=existing + value)
        if simple:
            components = nx.connected_components(simple)
            component = max(components, key=lambda nodes: (len(nodes), sorted(nodes)))
            return simple.subgraph(component).copy()
        return simple
    if path.suffix.lower() != ".json":
        raise ValueError("Use a GraphML file or EAGLE scenario JSON")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Expected an EAGLE schema version 1 scenario")
    graph = nx.Graph()
    graph.add_nodes_from(row["id"] for row in data["switches"])
    for link in data["links"]:
        a, b = link["source"], link["target"]
        if a in graph and b in graph:
            graph.add_edge(a, b, weight=float(link.get("partition_weight", 1)))
    if graph and not nx.is_connected(graph):
        raise ValueError("A scenario's switch graph must be connected for placement")
    return graph


def export_scenario(
    template: str | Path,
    assignments: list[dict[str, Any]],
    output: str | Path,
) -> None:
    """Assign one existing controller and its supporting switch to each community."""
    template = Path(template).resolve()
    output = Path(output).resolve()
    scenario = json.loads(template.read_text())
    controllers = scenario["controllers"]
    if len(assignments) != len(controllers):
        raise ValueError("The template needs one controller per partition")
    expected_nodes = {switch["id"] for switch in scenario["switches"]}
    if {node for group in assignments for node in group["switches"]} != expected_nodes:
        raise ValueError("Template switch IDs differ from the partition")
    slots = {row["support_switch"]: row for row in controllers}
    owners = {}
    for group in assignments:
        matches = slots.keys() & set(group["switches"])
        if len(matches) != 1:
            raise ValueError(
                "Each community must contain exactly one template supporting switch"
            )
        controller = slots[next(iter(matches))]
        controller["placement_switch"] = group["placement_switch"]
        for node in group["switches"]:
            owners[node] = controller["id"]
    for switch in scenario["switches"]:
        switch["controller"] = owners[switch["id"]]
    for profile in scenario.get("traffic", {}).values():
        for job in profile:
            if job["kind"] == "pcap":
                job["path"] = os.path.relpath(
                    template.parent / job["path"], output.parent
                )
    scenario["id"] += "-partitioned"
    scenario["provenance"] = f"Controller assignment generated from {template.name}"
    output.write_text(json.dumps(scenario, indent=2) + "\n")
