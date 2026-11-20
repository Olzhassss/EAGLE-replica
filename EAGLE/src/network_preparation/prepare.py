"""Generate seeded EAGLE networks from random graphs or bundled Topology Zoo files."""

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from ipaddress import IPv4Address
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import networkx as nx

from EAGLE.src import config
from EAGLE.src.data_preparation.hosts import dataset_hosts
from EAGLE.src.network_preparation.balanced_louvain import partition_domains
from EAGLE.src.network_preparation.graphs import load_switch_graph, random_switch_graph
from EAGLE.src.network_preparation.placement import (
    assign_hosts,
    place_controllers,
    select_controllable,
)
from EAGLE.src.network_preparation.plots import save_network_plot
from EAGLE.src.paths import ROOT, SOURCE_NETWORK
from network_schema import (
    NETWORK_SCHEMA_VERSION,
    resolve_support_counts,
    validate_support_layout,
)


TOPOLOGY_ZOO = Path(__file__).with_name("topology-zoo")
BPS_PER_MBPS = 1_000_000


def build_network(
    name: str,
    graph: nx.Graph,
    hosts: int,
    source: str,
    seed: int | None = None,
    *,
    support_switches: int | Sequence[int] | None = None,
) -> dict:
    """Allocate controller slots, controllable switches, host IDs and ports."""
    domains = config.N_DOMAINS
    counts = resolve_support_counts(
        config.SWITCHES_PER_DOMAIN if support_switches is None else support_switches,
        domains,
    )
    if not 1 <= domains <= len(graph):
        raise ValueError(f"Need at least {domains} switches for the configured domains")
    if hosts < 0:
        raise ValueError("Host count must be nonnegative")
    if (
        graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
        or not nx.is_connected(graph)
    ):
        raise ValueError("Use a connected, simple, undirected switch graph")

    graph = graph.copy()
    seed = config.SEED if seed is None else seed
    rng = Random(seed)
    switch_links = sorted(tuple(sorted(edge)) for edge in graph.edges)
    for a, b in switch_links:
        graph[a][b]["capacity_bps"] = rng.randint(58, 100) * BPS_PER_MBPS
    capacity = config.LOUVAIN_CAPACITY
    options = {
        "capacity": math.ceil(len(graph) / domains) if capacity is None else capacity,
        "difference": config.LOUVAIN_DIFFERENCE,
        "rho": config.LOUVAIN_RHO,
        "resolution": config.LOUVAIN_RESOLUTION,
        "seed": seed,
        "weight": "capacity_bps",
    }
    pre_merge, communities = partition_domains(graph, domains, **options)
    placements = place_controllers(graph, communities)
    groups = [set(placement["switches"]) for placement in placements]
    controllable = select_controllable(graph, groups, counts)
    access = assign_hosts(graph, groups, controllable, hosts)

    owners = {node: f"c{index}" for index, group in enumerate(groups) for node in group}
    controllers = [
        {
            "id": f"c{index}",
            "host": "127.0.0.1",
            "openflow_port": 6633 + index,
            "http_port": max(8080, 6633 + domains) + index,
            "placement_switch": f"s{placement['placement_switch']}",
            "support_switches": [f"s{node}" for node in reserved],
            "mean_hops": placement["mean_hops"],
            "initial_action": (1 << len(reserved)) - 1,
        }
        for index, (placement, reserved) in enumerate(
            zip(placements, controllable, strict=True)
        )
    ]
    switches = []
    for node in sorted(graph):
        switch = {"id": f"s{node}", "dpid": node, "controller": owners[node]}
        if "source_node" in graph.nodes[node]:
            switch["source_node"] = graph.nodes[node]["source_node"]
        switches.append(switch)
    base_address = IPv4Address("10.0.0.0")
    host_rows = [
        {
            "id": f"h{index}",
            "ip": f"{base_address + index}/8",
            "mac": (b"\x02\x00" + index.to_bytes(4, "big")).hex(":"),
        }
        for index in range(1, hosts + 1)
    ]

    ports = defaultdict(int)
    links = []

    def add_link(a: int, target: str, target_port: int, capacity_bps: int) -> None:
        ports[a] += 1
        links.append(
            {
                "source": f"s{a}",
                "target": target,
                "source_port": ports[a],
                "target_port": target_port,
                "capacity_bps": capacity_bps,
            }
        )

    for a, b in switch_links:
        ports[b] += 1
        add_link(a, f"s{b}", ports[b], graph[a][b]["capacity_bps"])
    for node, host in zip(access, host_rows, strict=True):
        add_link(node, host["id"], 0, rng.randint(58, 100) * BPS_PER_MBPS)

    return {
        "schema_version": NETWORK_SCHEMA_VERSION,
        "id": name,
        "provenance": {
            "source": source,
            "partition": options,
            "pre_merge_communities": [
                [f"s{node}" for node in sorted(group)] for group in pre_merge
            ],
        },
        "controllers": controllers,
        "switches": switches,
        "hosts": host_rows,
        "links": links,
        "telemetry": {
            "poll_seconds": 1,
            "peer_poll_seconds": 0.3,
            "peer_timeout_seconds": 2,
        },
    }


def use_dataset_hosts(network: dict, hosts: list[dict]) -> dict:
    """Keep topology attachments and take host identities from prepared traffic."""
    if len(hosts) != len(network["hosts"]):
        raise ValueError(f"{network['id']} needs {len(network['hosts'])} dataset hosts")
    names = {
        old["id"]: new["id"] for old, new in zip(network["hosts"], hosts, strict=True)
    }
    for link in network["links"]:
        for endpoint in ("source", "target"):
            link[endpoint] = names.get(link[endpoint], link[endpoint])
    network["hosts"] = hosts
    return network


def source_network(
    variant: str,
    hosts: list[dict],
    seed: int = config.SEED,
    *,
    support_switches: int | Sequence[int] | None = None,
) -> dict:
    """Build V0 from its source graph, or its reproducible 16-switch V3 extension."""
    network = json.loads(SOURCE_NETWORK.read_text())
    network["id"] = variant
    network["provenance"] = {"source": "source_v0.json", "seed": seed}
    if variant not in ("v0", "v3"):
        raise ValueError("Source variants are v0 and v3")
    if variant == "v3":
        rng = Random(seed)
        ports = defaultdict(int)
        for link in network["links"]:
            for endpoint in ("source", "target"):
                ports[link[endpoint]] = max(
                    ports[link[endpoint]], link[endpoint + "_port"]
                )
        additions = (
            (14, "c0", ("s1", "s6")),
            (15, "c2", ("s3", "s7")),
            (16, "c1", ("s5", "s9")),
        )
        for number, owner, neighbors in additions:
            node = f"s{number}"
            network["switches"].append({"id": node, "dpid": number, "controller": owner})
            for neighbor in neighbors:
                ports[node] += 1
                ports[neighbor] += 1
                network["links"].append(
                    {
                        "source": node,
                        "target": neighbor,
                        "source_port": ports[node],
                        "target_port": ports[neighbor],
                        "capacity_bps": rng.randint(58, 100) * BPS_PER_MBPS,
                    }
                )
    graph = nx.Graph()
    graph.add_nodes_from(row["id"] for row in network["switches"])
    graph.add_edges_from(
        (row["source"], row["target"])
        for row in network["links"]
        if row["source"] in graph and row["target"] in graph
    )
    groups = []
    for controller in network["controllers"]:
        group = {
            row["id"]
            for row in network["switches"]
            if row["controller"] == controller["id"]
        }
        controller.update(place_controllers(graph, [group])[0])
        controller.pop("switches")
        groups.append(group)
    counts = resolve_support_counts(
        config.SWITCHES_PER_DOMAIN if support_switches is None else support_switches,
        len(groups),
    )
    if counts != tuple(len(row["support_switches"]) for row in network["controllers"]):
        host_ids = {row["id"] for row in network["hosts"]}
        access = {
            link[endpoint]
            for link in network["links"]
            for endpoint, other in (("source", "target"), ("target", "source"))
            if link[other] in host_ids
        }
        chosen = select_controllable(graph, groups, counts, excluded=access)
        for controller, nodes in zip(network["controllers"], chosen, strict=True):
            controller["support_switches"] = nodes
            controller["initial_action"] = (1 << len(nodes)) - 1
    network["provenance"]["pre_merge_communities"] = [sorted(group) for group in groups]
    return use_dataset_hosts(network, hosts)


def write_network(network: dict, output: Path) -> Path:
    """Stage the JSON definition and both network plots before replacing outputs."""
    validate_support_layout(network)
    output.mkdir(parents=True, exist_ok=True)
    name = network["id"]
    with TemporaryDirectory(prefix=".network-", dir=output) as temporary:
        document = Path(temporary) / f"{name}.json"
        plot = document.with_suffix(".svg")
        pre_merge_plot = document.with_name(f"{name}-pre-merge.svg")
        document.write_text(json.dumps(network, indent=2) + "\n")
        save_network_plot(network, plot, config.SEED)
        save_network_plot(network, pre_merge_plot, config.SEED, pre_merge=True)
        pre_merge_plot.replace(output / pre_merge_plot.name)
        plot.replace(output / plot.name)
        return document.replace(output / document.name)


def main() -> None:
    """Generate source variants, random networks, or normalized Topology Zoo networks."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    random = modes.add_parser("random", help="Generate a connected random switch graph")
    zoo = modes.add_parser("zoo", help="Prepare the bundled Topology Zoo switch graphs")
    source_modes = [modes.add_parser(name) for name in ("v0", "v3")]
    for mode in (random, zoo, *source_modes):
        mode.add_argument("--output-dir", default=ROOT / "networks", type=Path)
        mode.add_argument("--seed", type=int, default=config.SEED)
        mode.add_argument(
            "--support-switches",
            type=int,
            nargs="+",
            help="Support switches per domain: one count for all domains, or counts in controller order",
        )
    for mode in (random, zoo):
        inputs = mode.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--hosts", type=int)
        inputs.add_argument("--dataset", type=Path)
        mode.add_argument("--name", help="Output name without .json")
    for mode in source_modes:
        mode.add_argument("--dataset", required=True, type=Path)
    random.add_argument("--nodes", required=True, type=int, help="Number of switches")
    random.add_argument(
        "--links", required=True, type=int, help="Number of switch-to-switch links"
    )
    zoo.add_argument(
        "--topo", choices=sorted(path.stem for path in TOPOLOGY_ZOO.glob("*.graphml"))
    )
    args = parser.parse_args()

    if args.mode in ("v0", "v3"):
        network = source_network(
            args.mode,
            dataset_hosts(args.dataset),
            args.seed,
            support_switches=args.support_switches,
        )
        print(write_network(network, args.output_dir))
        return
    hosts = dataset_hosts(args.dataset) if args.dataset else None
    count = len(hosts) if hosts else args.hosts
    if args.name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise ValueError("Use a filename stem for --name")
    if args.mode == "zoo" and args.name and not args.topo:
        raise ValueError("--name requires one --topo")
    if args.mode == "random":
        graph = random_switch_graph(args.nodes, args.links, args.seed)
        name = f"random-{args.nodes}n-{args.links}l"
        graphs = [(name, name, graph)]
    else:
        paths = (
            [TOPOLOGY_ZOO / f"{args.topo}.graphml"]
            if args.topo
            else sorted(TOPOLOGY_ZOO.glob("*.graphml"))
        )
        graphs = [(path.stem, path.name, load_switch_graph(path)) for path in paths]
    for name, source, graph in graphs:
        network = build_network(
            args.name or f"{name}-{count}hosts",
            graph,
            count,
            source,
            args.seed,
            support_switches=args.support_switches,
        )
        if hosts:
            use_dataset_hosts(network, hosts)
        print(write_network(network, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
