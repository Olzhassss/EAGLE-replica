"""Compare three trained models across source, traffic-load, and congestion datasets."""

import argparse

from .artifacts import EVAL, write_json
from .cli import add_evaluation_arguments, artifact_name, load_network
from .datasets import DATASET_GROUPS, load_dataset
from .evaluate import evaluate_condition
from .plots import compare_condition, plot_bars
from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--name", type=artifact_name, default="datasets")
    args = parser.parse_args()
    try:
        network = load_network("v0.json")
        settings = load_settings(args.settings)
        datasets = {
            name: load_dataset(network, name, "test")
            for names in DATASET_GROUPS.values()
            for name in names
        }
        conditions = []
        for group, names in DATASET_GROUPS.items():
            rows = []
            for name in names:
                condition = evaluate_condition(
                    args, network, settings, datasets[name], args.name
                )
                rows.append(compare_condition(condition, args.models, args.seed))
                conditions.append(rows[-1])
                write_json(
                    EVAL / f"{args.name}.json",
                    {
                        "models": args.models,
                        "seed": args.seed,
                        "conditions": conditions,
                    },
                )
            plot_bars(rows, args.models, EVAL / f"{args.name}_{group}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
