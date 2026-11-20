"""Plot the first nonzero all-on observation from each traffic condition on V0."""

import argparse

from EAGLE.src.arguments import add_runtime_arguments, artifact_name
from EAGLE.src.artifacts import write_json
from EAGLE.src.datasets import DATASET_GROUPS, load_dataset
from EAGLE.src.evaluation.embeddings import collect_embeddings
from EAGLE.src.evaluation.plots import DATASET_LABELS, plot_embeddings
from EAGLE.src.network import load_network
from EAGLE.src.paths import EVAL
from EAGLE.src.settings import load_settings


def main() -> None:
    """Collect traffic-bearing observations from each condition and project them together."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--model", type=artifact_name, required=True)
    parser.add_argument("--name", type=artifact_name, default="traffic_embeddings")
    args = parser.parse_args()
    network = load_network("v0.json")
    settings = load_settings(args.settings)
    names = (
        "dc_t0",
        "dc_t1",
        "mg_t0",
        "dc_t3",
        "dc_t4",
        *DATASET_GROUPS["congestions"],
    )
    datasets = [load_dataset(network, name, "test") for name in names]
    records = [
        collect_embeddings(args, network, settings, dataset, 1, nonzero=True)
        for dataset in datasets
    ]
    write_json(EVAL / f"{args.name}.json", {"conditions": records})
    plot_embeddings(
        [
            {
                "label": DATASET_LABELS[record["dataset"]],
                "embedding": record["embeddings"][0],
            }
            for record in records
        ],
        EVAL / args.name,
        args.seed,
    )


if __name__ == "__main__":
    main()
