"""Collect 16 all-on observations per topology and plot V0 through V4 together."""

import argparse
import json

from .artifacts import EVAL, write_json
from .cli import add_runtime_arguments, artifact_name, load_network
from .datasets import load_dataset
from .embeddings import collect_embeddings
from .eval_topologies import TOPOLOGIES
from .plots import plot_embeddings
from .settings import load_settings


def plot_collected(args) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--model", type=artifact_name, required=True)
    parser.add_argument("--network", choices=[name + ".json" for name in TOPOLOGIES])
    parser.add_argument("--name", type=artifact_name, default="topology_embeddings")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    try:
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
        if all(
            (EVAL / f"{args.name}_{topology}.json").exists() for topology in TOPOLOGIES
        ):
            plot_collected(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
