"""Arguments shared by the training and evaluation entrypoints."""

import argparse
from pathlib import Path
from typing import Any

import torch

from .scenario import DEFAULT_SCENARIO, Scenario, load_scenario
from .settings import DEFAULT_SETTINGS, load_settings


def add_trial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument(
        "--mininet-socket", type=Path, help="Override the address in mininet_nets/.socket"
    )
    parser.add_argument(
        "--ryu-socket", type=Path, help="Override the address in the repository's .socket"
    )
    parser.add_argument("--traffic", default="idle", help="Scenario traffic profile")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resource-mode", choices=("forwarding", "interfaces", "ovs"))
    parser.add_argument("--keep-captures", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for the EAGLE models: auto, cpu, cuda, or cuda:N",
    )


def resolve_device(requested: str) -> torch.device:
    """Select the model device and validate explicit CUDA requests."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(
            f"Invalid device {requested!r}. Use auto, cpu, cuda, or cuda:N",
        ) from error
    if device == torch.device("cpu"):
        return device
    if device.type != "cuda":
        raise ValueError(
            f"Unsupported device {requested!r}. Use auto, cpu, cuda, or cuda:N",
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is unavailable to this PyTorch runtime"
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device.index} is outside the visible device range"
        )
    return device


def parse_trial_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    normalization: str | None = None,
) -> tuple[Scenario, dict[str, Any]]:
    if args.episodes <= 0 or args.seed < 0:
        parser.error("Episodes must be positive and seed must be nonnegative")
    try:
        scenario = load_scenario(args.scenario)
        settings = load_settings(
            args.settings,
            episode_steps=args.episode_steps,
            resource_mode=args.resource_mode,
            normalization=normalization,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return scenario, settings
