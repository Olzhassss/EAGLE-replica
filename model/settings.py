"""Validated training and evaluation settings."""

import copy
import json
import math
from pathlib import Path

from .scenario import ROOT

DEFAULT_SETTINGS = ROOT / "settings/default.json"


def positive(value: object, name: str, *, integer: bool = False, zero: bool = False) -> None:
    expected = (int,) if integer else (int, float)
    if (
        type(value) not in expected
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}")


def load_settings(path: Path = DEFAULT_SETTINGS) -> dict:
    defaults = json.loads(DEFAULT_SETTINGS.read_text())
    updates = json.loads(Path(path).read_text())
    data = copy.deepcopy(defaults)

    def merge(target: dict, source: dict) -> None:
        for key, value in source.items():
            if key not in target:
                raise ValueError(f"Unknown setting {key}")
            if isinstance(value, dict) and isinstance(target[key], dict):
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


def training_stages(mode: str, episodes: int, pretrain_episodes: int) -> list[dict]:
    """Pretrain across the source conditions, then freeze the estimator."""
    positive(episodes, "episodes", integer=True)
    positive(pretrain_episodes, "pretrain_episodes", integer=True)
    profiles = ["dc_pcap"] if mode == "pcaps" else ["dc_t0"]
    if mode == "curriculum":
        profiles += [f"dc_t0_trenv{index}" for index in range(2, 6)]
    if mode not in ("pcaps", "synthetic", "curriculum"):
        raise ValueError("Training mode must be pcaps, synthetic, or curriculum")
    if episodes < len(profiles):
        raise ValueError("Curriculum training needs at least five policy episodes")
    stages = [{
        "name": "pretrain", "mode": "pretrain",
        "profiles": profiles, "episodes": pretrain_episodes,
    }]
    share, extra = divmod(episodes, len(profiles))
    stages.extend(
        {"name": name, "mode": "frozen", "profiles": [name], "episodes": share + (index < extra)}
        for index, name in enumerate(profiles)
    )
    return stages
