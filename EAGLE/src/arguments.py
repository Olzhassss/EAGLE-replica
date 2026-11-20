"""Shared arguments and device selection for the experiment entry points."""

import argparse
import re
from pathlib import Path

import torch

from EAGLE.src import config
from EAGLE.src.paths import DEFAULT_SETTINGS


def artifact_name(value: str) -> str:
    """Parse a model or experiment name that stays within its output directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (
        "checkpoints",
        "runs",
    ):
        raise argparse.ArgumentTypeError(
            "Use a model or experiment name without directories"
        )
    return value


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add settings, seed, socket, and device options shared by experiments."""
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--mininet-socket", type=Path)
    parser.add_argument("--ryu-socket", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the three model names and repetition budget to runtime options."""
    add_runtime_arguments(parser)
    parser.add_argument(
        "--models",
        nargs=3,
        type=artifact_name,
        default=["dc_t0", "dc_pcaps", "dc_full"],
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--repetitions", type=int)


def validate_models(models: list[str]) -> None:
    """Keep three learned policies distinct from each other and the all-on baseline."""
    if len(models) != 3 or len(set(models)) != 3 or "baseline" in models:
        raise ValueError("Choose three distinct model names; baseline is reserved")


def resolve_device(requested: str) -> torch.device:
    """Choose CPU or a CUDA device, resolving auto against the current machine."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cpu":
        return device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Select CPU or an available CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} is unavailable")
    return device
