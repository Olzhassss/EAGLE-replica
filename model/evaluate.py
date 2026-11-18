"""Shared frozen-policy evaluation and matched baseline measurements."""

import argparse
import random

import numpy as np
import torch

from .artifacts import EVAL, MODELS
from .checkpoints import checkpoint_digest, load_inference_model
from .cli import resolve_device
from .controller_client import ControllerClient
from .datasets import Dataset
from .emulation import Emulation
from .joint_environment import JointEnvironment
from .observations import ObservationEncoder
from .run_log import RunLog
from .scenario import Scenario
from .settings import positive
from .trials import run_trial


def seed_everything(seed: int) -> None:
    positive(seed, "seed", integer=True, zero=True)
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)


def make_environment(
    args: argparse.Namespace, network: Scenario, settings: dict, encoder: ObservationEncoder,
) -> JointEnvironment:
    return JointEnvironment(
        network, settings, ControllerClient(network),
        Emulation(args.mininet_socket, args.ryu_socket), encoder,
    )


def evaluate_condition(
    args: argparse.Namespace, network: Scenario, settings: dict, dataset: Dataset, suite: str,
) -> dict:
    """Run a shared baseline and three models with seed + repetition index."""
    episodes = settings["evaluation"]["episodes"] if args.episodes is None else args.episodes
    repetitions = settings["evaluation"]["repetitions"] if args.repetitions is None else args.repetitions
    positive(episodes, "episodes", integer=True)
    positive(repetitions, "repetitions", integer=True)
    if len(set(args.models)) != 3:
        raise ValueError("Specify three distinct model names")
    loaded = {name: load_inference_model(MODELS / name, network) for name in args.models}
    device = resolve_device(args.device)
    for policy, _ in loaded.values():
        policy.to(device)
    encoder = next(iter(loaded.values()))[1]
    methods = {"baseline": (lambda _: [1] * len(network.controller_ids), encoder)}
    methods.update({
        name: (policy.decide, saved_encoder)
        for name, (policy, saved_encoder) in loaded.items()
    })
    measurements = {name: [] for name in methods}
    for repetition in range(repetitions):
        seed = args.seed + repetition
        for name, (decide, saved_encoder) in methods.items():
            seed_everything(seed)
            environment = make_environment(args, network, settings, saved_encoder)
            directory = EVAL / "runs" / suite / network.path.stem / dataset.name / name / f"seed-{seed}"
            log = RunLog(
                directory, network, settings, overwrite=True, seed=seed,
                model=name, dataset=dataset.name, encoder=saved_encoder.to_dict(),
            )
            results = []
            for episode in range(1, episodes + 1):
                print(
                    f"{network.path.stem} | {dataset.name} | {name} | "
                    f"repetition {repetition + 1}/{repetitions} | episode {episode}/{episodes}",
                    flush=True,
                )
                results.append(run_trial(environment, decide, log, episode=episode, dataset=dataset))
            count = sum(row["completed_flows"] for row in results)
            measured = sum(row["measured_seconds"] for row in results)
            total_fct = sum(
                row["fct_seconds"] * row["completed_flows"]
                for row in results if row["fct_seconds"] is not None
            )
            energy = sum(row["power_watts"] * row["measured_seconds"] for row in results)
            measurements[name].append({
                "seed": seed,
                "fct_seconds": total_fct / count if count else None,
                "power_watts": energy / measured,
                "episode_rewards": [row["cumulative_reward"] for row in results],
            })
    return {
        "dataset": dataset.name, "dataset_sha256": dataset.fingerprint,
        "models_sha256": {name: checkpoint_digest(MODELS / name) for name in args.models},
        "settings": settings, "base_seed": args.seed,
        "episodes": episodes, "repetitions": repetitions, "measurements": measurements,
    }
