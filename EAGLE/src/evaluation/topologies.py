"""Collect one manually started topology per invocation, then compare V0 through V4."""

import argparse
import json

import numpy as np

from EAGLE.src.arguments import add_evaluation_arguments, artifact_name
from EAGLE.src.artifacts import save_rewards, write_json
from EAGLE.src.datasets import load_dataset
from EAGLE.src.evaluation.plots import compare_condition, plot_bars
from EAGLE.src.evaluation.policy_comparison import evaluate_condition
from EAGLE.src.network import load_network
from EAGLE.src.paths import EVAL
from EAGLE.src.settings import load_settings

TOPOLOGIES = tuple(f"v{index}" for index in range(5))


def plot_collected(name: str, models: list[str]) -> None:
    """Compare saved topology results after checking their experiment identities."""
    conditions = []
    for topology in TOPOLOGIES:
        data = json.loads((EVAL / f"{name}_{topology}.json").read_text())
        if data["models"] != models:
            raise ValueError("Collected topologies use different model lists")
        conditions.append(data["condition"])
    if any(
        condition[key] != conditions[0][key]
        for condition in conditions[1:]
        for key in ("models_sha256", "settings", "base_seed", "episodes", "repetitions")
    ):
        raise ValueError(
            "Use the same models, settings, seed, and repetition schedule across topologies"
        )
    plot_bars(conditions, models, EVAL / name, topologies=True)


def main() -> None:
    """Collect the running topology or redraw the aggregate topology comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--network", choices=[name + ".json" for name in TOPOLOGIES])
    parser.add_argument("--name", type=artifact_name, default="topologies")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_collected(args.name, args.models)
        return
    if args.network is None:
        parser.error("Supply --network for the currently running topology")
    network = load_network(args.network)
    settings = load_settings(args.settings)
    dataset = load_dataset(network, "dc_t0", "test")
    condition = compare_condition(
        evaluate_condition(args, network, settings, dataset, args.name),
        args.models,
        args.seed,
    )
    condition["network"] = network.path.stem
    for model in args.models:
        rewards = np.mean(
            [row["episode_rewards"] for row in condition["measurements"][model]],
            axis=0,
        ).tolist()
        save_rewards(EVAL / f"{model}_{network.path.stem}.csv", rewards)
    write_json(
        EVAL / f"{args.name}_{network.path.stem}.json",
        {"models": args.models, "seed": args.seed, "condition": condition},
    )
    if all((EVAL / f"{args.name}_{topology}.json").exists() for topology in TOPOLOGIES):
        plot_collected(args.name, args.models)


if __name__ == "__main__":
    main()
