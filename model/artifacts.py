"""Atomic experiment data and reward curves."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .scenario import ROOT

MODELS = ROOT / "models"
EVAL = ROOT / "eval"


def write_json(path: Path, data: dict) -> None:
    atomic_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".write-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_rewards(path: Path, rewards: list[float], *, plot: bool = True) -> None:
    from io import StringIO

    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(("episode", "cumulative_reward"))
    writer.writerows(enumerate(rewards, 1))
    atomic_text(path, stream.getvalue())
    if plot:
        figure = Figure(figsize=(8, 4), layout="constrained")
        axis = figure.subplots()
        axis.plot(range(1, len(rewards) + 1), rewards, marker=".")
        axis.set(xlabel="Episode", ylabel="Cumulative reward", title=path.stem)
        axis.grid(alpha=0.25)
        figure.savefig(path.with_suffix(".png"), dpi=160)


def read_rewards(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    episodes = np.array([int(row["episode"]) for row in rows])
    rewards = np.array([float(row["cumulative_reward"]) for row in rows])
    if (
        not len(rows)
        or not np.isfinite(rewards).all()
        or len(set(episodes)) != len(episodes)
        or (np.diff(episodes) <= 0).any()
    ):
        raise ValueError(
            f"Expected finite rewards in increasing episode order in {path}"
        )
    return episodes, rewards
