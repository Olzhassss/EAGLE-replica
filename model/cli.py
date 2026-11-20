"""Shared arguments and device selection for the experiment entry points."""

import argparse
import re
from pathlib import Path

import torch

from config import Config
from .scenario import ROOT, Scenario, load_scenario


def artifact_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (
        "checkpoints",
        "runs",
    ):
        raise argparse.ArgumentTypeError(
            "Use a model or experiment name without directories"
        )
    return value


def network_file(value: str) -> Path:
    if Path(value).name != value or not value.endswith(".json"):
        raise ValueError("Specify a JSON filename from networks/")
    return ROOT / "networks" / value


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settings", type=Path, default=ROOT / "settings/default.json")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--mininet-socket", type=Path)
    parser.add_argument("--ryu-socket", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    add_runtime_arguments(parser)
    parser.add_argument(
        "--models",
        nargs=3,
        type=artifact_name,
        default=["dc_t0", "dc_pcaps", "dc_full"],
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--repetitions", type=int)


def load_network(value: str) -> Scenario:
    return load_scenario(network_file(value))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(f"Invalid device {requested!r}") from error
    if device.type == "cpu":
        return device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Select CPU or an available CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} is unavailable")
    return device
