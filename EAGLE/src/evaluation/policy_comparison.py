"""Shared frozen-policy evaluation and matched baseline measurements."""

import argparse

from EAGLE.src.arguments import resolve_device, validate_models
from EAGLE.src.datasets import Dataset
from EAGLE.src.experiments.run_log import RunLog
from EAGLE.src.experiments.runtime import make_environment, seed_everything
from EAGLE.src.experiments.trials import run_trial
from EAGLE.src.model.checkpoints import checkpoint_digest, load_inference_model
from EAGLE.src.network import Network
from EAGLE.src.paths import EVAL, MODELS
from EAGLE.src.settings import positive


def load_policies(models: list[str], network: Network, device: str) -> dict:
    """Load frozen policies and pair each decision function with its saved encoder."""
    loaded = {name: load_inference_model(MODELS / name, network) for name in models}
    selected_device = resolve_device(device)
    encoder = next(iter(loaded.values()))[1]
    methods = {"baseline": (lambda _: network.all_on_actions, encoder)}
    for name, (policy, saved_encoder) in loaded.items():
        policy.to(selected_device)
        methods[name] = policy.decide, saved_encoder
    return methods


def evaluate_condition(
    args: argparse.Namespace,
    network: Network,
    settings: dict,
    dataset: Dataset,
    suite: str,
) -> dict:
    """Run a shared baseline and three models with seed + repetition index."""
    episodes = (
        settings["evaluation"]["episodes"] if args.episodes is None else args.episodes
    )
    repetitions = (
        settings["evaluation"]["repetitions"]
        if args.repetitions is None
        else args.repetitions
    )
    positive(episodes, "episodes", integer=True)
    positive(repetitions, "repetitions", integer=True)
    positive(args.seed, "seed", integer=True, zero=True)
    validate_models(args.models)
    methods = load_policies(args.models, network, args.device)
    measurements = {name: [] for name in methods}
    for repetition in range(repetitions):
        seed = args.seed + repetition
        for name, (decide, saved_encoder) in methods.items():
            seed_everything(seed)
            environment = make_environment(args, network, settings, saved_encoder)
            directory = (
                EVAL
                / "runs"
                / suite
                / network.path.stem
                / dataset.name
                / name
                / f"seed-{seed}"
            )
            log = RunLog(
                directory,
                network,
                settings,
                overwrite=True,
                seed=seed,
                model=name,
                dataset=dataset.name,
                encoder=saved_encoder.to_dict(),
            )
            results = []
            for episode in range(1, episodes + 1):
                print(
                    f"{network.path.stem} | {dataset.name} | {name} | "
                    f"repetition {repetition + 1}/{repetitions} | episode {episode}/{episodes}",
                    flush=True,
                )
                results.append(
                    run_trial(environment, decide, log, episode=episode, dataset=dataset)
                )
            count = sum(row["completed_flows"] for row in results)
            measured = sum(row["measured_seconds"] for row in results)
            total_fct = sum(
                row["fct_seconds"] * row["completed_flows"]
                for row in results
                if row["fct_seconds"] is not None
            )
            energy = sum(row["power_watts"] * row["measured_seconds"] for row in results)
            measurements[name].append(
                {
                    "seed": seed,
                    "fct_seconds": total_fct / count if count else None,
                    "power_watts": energy / measured,
                    "episode_rewards": [row["cumulative_reward"] for row in results],
                }
            )
    return {
        "dataset": dataset.name,
        "dataset_sha256": dataset.fingerprint,
        "models_sha256": {name: checkpoint_digest(MODELS / name) for name in args.models},
        "settings": settings,
        "base_seed": args.seed,
        "episodes": episodes,
        "repetitions": repetitions,
        "measurements": measurements,
    }
