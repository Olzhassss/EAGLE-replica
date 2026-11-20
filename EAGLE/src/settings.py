"""Validated training and evaluation settings."""

import json
import math
from pathlib import Path

from EAGLE.src.paths import DEFAULT_SETTINGS


def positive(
    value: object, name: str, *, integer: bool = False, zero: bool = False
) -> None:
    """Reject nonfinite or out-of-range values used as counts and time intervals."""
    expected = (int,) if integer else (int, float)
    if (
        type(value) not in expected
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(
            f"{name} must be a {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}"
        )


def load_settings(path: Path = DEFAULT_SETTINGS) -> dict:
    """Merge a partial settings file into defaults and check numerical constraints."""
    data = json.loads(DEFAULT_SETTINGS.read_text())
    updates = json.loads(Path(path).read_text())

    def merge(target: dict, source: dict) -> None:
        if not isinstance(source, dict):
            raise ValueError("Settings must be a JSON object")
        for key, value in source.items():
            if key not in target:
                raise ValueError(f"Unknown setting {key}")
            if isinstance(target[key], dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(data, updates)
    for key in ("decision_seconds", "sample_timeout_seconds"):
        positive(data[key], key)
    for key in ("warmup_seconds", "drain_seconds"):
        positive(data[key], key, zero=True)
    for key in ("order", "eval_points"):
        positive(data["embedding"][key], "embedding." + key, integer=True)
    if data["embedding"]["normalization"] not in ("source", "identity"):
        raise ValueError("embedding.normalization must be source or identity")
    reward = data["reward"]
    positive(reward["alpha"], "reward.alpha", zero=True)
    for key in ("throughput_reference_bps", "power_reference_watts"):
        positive(reward[key], "reward." + key)
    for key, value in reward["power_model"].items():
        positive(value, "power_model." + key, zero=True)
    for key in ("episodes", "pretrain_episodes", "calibration_episodes"):
        positive(data["training"][key], "training." + key, integer=True)
    learning = data["learning"]
    for key in ("actor_lr", "critic_lr", "estimator_lr"):
        positive(learning[key], "learning." + key)
    positive(learning["batch_size"], "learning.batch_size", integer=True)
    pruning = learning["pruning"]
    if pruning["mode"] not in ("none", "static", "dynamic"):
        raise ValueError("pruning.mode must be none, static, or dynamic")
    positive(pruning["warmup_rounds"], "pruning.warmup_rounds", integer=True, zero=True)
    positive(pruning["ramp_rounds"], "pruning.ramp_rounds", integer=True)
    for key in ("episodes", "repetitions"):
        positive(data["evaluation"][key], "evaluation." + key, integer=True)
    return data
