"""Collect 16 all-on observations per topology and plot V0 through V4 together."""

import argparse
import json

from EAGLE.src.arguments import add_runtime_arguments, artifact_name
from EAGLE.src.artifacts import write_json
from EAGLE.src.datasets import load_dataset
from EAGLE.src.evaluation.embeddings import collect_embeddings
from EAGLE.src.evaluation.plots import plot_embeddings
from EAGLE.src.evaluation.topologies import TOPOLOGIES
from EAGLE.src.network import load_network
from EAGLE.src.paths import EVAL
from EAGLE.src.settings import load_settings


def plot_collected(args) -> None:
    """Combine all saved topology embeddings using the same model, encoder, and seed."""
    records = [
        json.loads((EVAL / f"{args.name}_{topology}.json").read_text())
        for topology in TOPOLOGIES
    ]
    if any(
        record["model"] != args.model
        or record["model_sha256"] != records[0]["model_sha256"]
        or record["seed"] != args.seed
        or record["encoder"] != records[0]["encoder"]
        or len(record["embeddings"]) != 16
        for record in records
    ):
        raise ValueError(
            "Collect all five topologies with the same model, seed, and encoder"
        )
    samples = [
        {"label": record["network"], "embedding": vector}
        for record in records
        for vector in record["embeddings"]
    ]
    plot_embeddings(samples, EVAL / args.name, args.seed)


def main() -> None:
    """Collect embeddings for the running network or redraw the combined projection."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--model", type=artifact_name, required=True)
    parser.add_argument("--network", choices=[name + ".json" for name in TOPOLOGIES])
    parser.add_argument("--name", type=artifact_name, default="topology_embeddings")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_collected(args)
        return
    if args.network is None:
        parser.error("Supply --network for the currently running topology")
    network = load_network(args.network)
    settings = load_settings(args.settings)
    data = collect_embeddings(
        args, network, settings, load_dataset(network, "dc_t0", "test"), 16
    )
    write_json(EVAL / f"{args.name}_{network.path.stem}.json", data)
    if all((EVAL / f"{args.name}_{topology}.json").exists() for topology in TOPOLOGIES):
        plot_collected(args)


if __name__ == "__main__":
    main()
