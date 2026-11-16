"""Arguments shared by the training and evaluation entrypoints."""

import argparse
from pathlib import Path
from typing import Any

from .settings import (
    DEFAULT_SCENARIO,
    DEFAULT_SETTINGS,
    Scenario,
    load_scenario,
    load_settings,
)


def add_trial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--mininet-socket", type=Path)
    parser.add_argument("--ryu-socket", type=Path)
    parser.add_argument("--traffic", default="idle", help="Scenario traffic profile")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resource-mode", choices=("forwarding", "interfaces", "ovs"))
    parser.add_argument("--keep-captures", action="store_true")


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
