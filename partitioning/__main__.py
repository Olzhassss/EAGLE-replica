"""Partition a topology, place controllers, and optionally draw the communities."""

import argparse
import json
from pathlib import Path

from .balanced_louvain import balanced_partitions
from .placement import place_controllers
from .topology_io import TOPOLOGY_ZOO, export_scenario, load_switch_graph


def save_partition_plot(graph, placements, path, seed):
    import matplotlib.pyplot as plt
    import networkx as nx

    groups = {node: index for index, placement in enumerate(placements) for node in placement["switches"]}
    figure, axis = plt.subplots(figsize=(15, 15))
    nx.draw(graph, pos=nx.spring_layout(graph, seed=seed), ax=axis,
            node_color=[groups[node] for node in graph], cmap=plt.cm.rainbow,
            with_labels=True, node_size=500, font_color="white")
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--topo", help="Bundled Topology Zoo basename, default GtsCe")
    inputs.add_argument("--input", type=Path, help="GraphML or scenario JSON")
    parser.add_argument("--capacity", type=int, default=60)
    parser.add_argument("--difference", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Placement JSON path, default stdout")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plot", type=Path, help="Destination PNG")
    plots.add_argument("--no-plot", action="store_true")
    parser.add_argument("--scenario-template", type=Path)
    parser.add_argument("--export-scenario", type=Path)
    args = parser.parse_args()
    if bool(args.export_scenario) != bool(args.scenario_template):
        parser.error("Use --scenario-template with --export-scenario")
    topo = args.topo or "GtsCe"
    if args.input is None and topo not in {file.stem for file in TOPOLOGY_ZOO.glob("*.graphml")}:
        parser.error("Unknown bundled topology " + topo)
    path = args.input or TOPOLOGY_ZOO / (topo + ".graphml")
    try:
        graph = load_switch_graph(path)
        options = dict(capacity=args.capacity, difference=args.difference, rho=args.rho,
                       resolution=args.resolution, seed=args.seed)
        for partition in balanced_partitions(graph, **options):
            final = partition
        placements = place_controllers(graph, final)
        output = dict(schema_version=1, input=str(path.resolve()), switches=len(graph),
                      undirected_links=graph.number_of_edges(), communities=placements, parameters=options)
        document = json.dumps(output, indent=2) + "\n"
        if args.export_scenario:
            export_scenario(args.scenario_template, placements, args.export_scenario)
        if args.output:
            args.output.write_text(document)
        else:
            print(document, end="")
        if not args.no_plot:
            save_partition_plot(graph, placements, args.plot or Path(path.name + "_subnets.png"), args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
