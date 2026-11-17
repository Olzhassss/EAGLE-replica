"""Generate seeded EAGLE scenarios from random graphs or bundled Topology Zoo files."""

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from ipaddress import IPv4Address
from itertools import combinations
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import networkx as nx

from config import Config

from .balanced_louvain import partition_domains
from .placement import assign_hosts, place_controllers, select_controllable
from .plot import save_network_plot

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_ZOO = Path(__file__).with_name("topology-zoo")
BPS_PER_MBPS = 1_000_000


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


def zoo_graphs(paths: list[Path]) -> Iterator[tuple[Path, nx.Graph]]:
    """Read and report the switch and link counts used for each Zoo file."""
    for path in paths:
        graph = load_switch_graph(path)
        print(
            f"{path.stem}  {len(graph)} switches, {graph.number_of_edges()} links",
            flush=True,
        )
        yield path, graph


def build_scenario(name: str, graph: nx.Graph, hosts: int, source: str) -> dict:
    """Allocate controller slots, controllable switches, host IDs and ports."""
    domains = Config.N_DOMAINS
    per_domain = Config.SWITCHES_PER_DOMAIN
    if not 1 <= domains <= len(graph):
        raise ValueError(f"Need at least {domains} switches for the configured domains")
    if per_domain < 1:
        raise ValueError("Each domain needs at least one controllable switch")
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
    rng = Random(Config.SEED)
    switch_links = sorted(tuple(sorted(edge)) for edge in graph.edges)
    for a, b in switch_links:
        graph[a][b]["capacity_bps"] = rng.randint(58, 100) * BPS_PER_MBPS
    capacity = Config.LOUVAIN_CAPACITY
    options = {
        "capacity": math.ceil(len(graph) / domains) if capacity is None else capacity,
        "difference": Config.LOUVAIN_DIFFERENCE,
        "rho": Config.LOUVAIN_RHO,
        "resolution": Config.LOUVAIN_RESOLUTION,
        "seed": Config.SEED,
        "weight": "capacity_bps",
    }
    pre_merge, communities = partition_domains(graph, domains, **options)
    placements = place_controllers(graph, communities)
    groups = [set(placement["switches"]) for placement in placements]
    controllable = select_controllable(graph, groups, per_domain)
    access = assign_hosts(graph, groups, controllable, hosts)

    owners = {node: f"c{index}" for index, group in enumerate(groups) for node in group}
    controllers = [
        {
            "id": f"c{index}",
            "host": "127.0.0.1",
            "openflow_port": 6633 + index,
            "http_port": max(8080, 6633 + domains) + index,
            "placement_switch": f"s{placement['placement_switch']}",
            "support_switch": f"s{reserved[0]}",
            "controllable_switches": [f"s{node}" for node in reserved],
            "mean_hops": placement["mean_hops"],
            "initial_action": 1,
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
        "schema_version": 1,
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
        "traffic": {"idle": []},
    }


def write_network(scenario: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    name = scenario["id"]
    with TemporaryDirectory(prefix=".network-", dir=output) as temporary:
        document = Path(temporary) / f"{name}.json"
        plot = document.with_suffix(".svg")
        pre_merge_plot = document.with_name(f"{name}-pre-merge.svg")
        document.write_text(json.dumps(scenario, indent=2) + "\n")
        save_network_plot(scenario, plot, Config.SEED)
        save_network_plot(scenario, pre_merge_plot, Config.SEED, pre_merge=True)
        pre_merge_plot.replace(output / pre_merge_plot.name)
        plot.replace(output / plot.name)
        return document.replace(output / document.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    random = modes.add_parser("random", help="Generate a connected random switch graph")
    zoo = modes.add_parser("zoo", help="Prepare the bundled Topology Zoo switch graphs")
    for mode in (random, zoo):
        mode.add_argument("--hosts", required=True, type=int)
        mode.add_argument("--output-dir", default=ROOT / "networks", type=Path)
    random.add_argument("--nodes", required=True, type=int, help="Number of switches")
    random.add_argument(
        "--links", required=True, type=int, help="Number of switch-to-switch links"
    )
    zoo.add_argument(
        "--topo", choices=sorted(path.stem for path in TOPOLOGY_ZOO.glob("*.graphml"))
    )
    args = parser.parse_args()

    try:
        if args.mode == "random":
            graph = random_switch_graph(args.nodes, args.links, Config.SEED)
            name = f"random-{args.nodes}n-{args.links}l"
            graphs = [(name, name, graph)]
        else:
            paths = (
                [TOPOLOGY_ZOO / f"{args.topo}.graphml"]
                if args.topo
                else sorted(TOPOLOGY_ZOO.glob("*.graphml"))
            )
            graphs = (
                (path.stem, path.name, graph) for path, graph in zoo_graphs(paths)
            )
        for name, source, graph in graphs:
            try:
                scenario = build_scenario(
                    f"{name}-{args.hosts}hosts", graph, args.hosts, source
                )
                print(write_network(scenario, args.output_dir), flush=True)
            except ValueError as error:
                raise ValueError(f"{name}  {error}") from error
    except (OSError, ValueError, nx.NetworkXException) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
