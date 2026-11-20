"""Command dispatcher for data preparation, training, and evaluation."""

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "prepare-data": ("data_preparation.prepare", "Generate traffic datasets"),
    "prepare-network": ("network_preparation.prepare", "Generate network definitions"),
    "analyze-trace": ("data_preparation.analyze_trace", "Inspect packet captures"),
    "scale-traffic": ("data_preparation.synthetic", "Scale a synthetic dataset"),
    "train": ("training.train", "Train or resume a model"),
    "interactive": ("interactive", "Apply a frozen policy to manual traffic"),
    "eval-datasets": ("evaluation.datasets", "Compare policies across datasets"),
    "eval-topologies": ("evaluation.topologies", "Compare policies across networks"),
    "eval-switches": ("evaluation.switches", "Compare enabled switch counts"),
    "eval-pruning": ("evaluation.pruning", "Compare pruning reward curves"),
    "eval-topology-embeddings": (
        "evaluation.topology_embeddings",
        "Collect and plot network embeddings",
    ),
    "eval-traffic-embeddings": (
        "evaluation.traffic_embeddings",
        "Collect and plot traffic embeddings",
    ),
}


def main() -> None:
    """Dispatch a command lazily so help does not load the training libraries."""
    parser = argparse.ArgumentParser(prog="python -m EAGLE", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args()
    module, _ = COMMANDS[args.command]
    original = sys.argv
    sys.argv = [f"{parser.prog} {args.command}", *remaining]
    try:
        import_module(f"EAGLE.src.{module}").main()
    finally:
        sys.argv = original
