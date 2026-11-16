"""Load and validate run settings for the RL project."""

import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

from .scenario import ROOT

DEFAULT_SETTINGS = ROOT / "settings/default.json"


def load_settings(
    path: Path | str,
    *,
    episode_steps: int | None = None,
    resource_mode: str | None = None,
    normalization: str | None = None,
) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if episode_steps is not None:
        data["episode_steps"] = episode_steps
    if normalization is not None:
        data["embedding"]["normalization"] = normalization
    if resource_mode is not None:
        data["resource_mode"] = resource_mode
    data.setdefault("resource_mode", "forwarding")
    data.setdefault("failures", [])
    if data["resource_mode"] not in ("forwarding", "interfaces", "ovs"):
        raise ValueError("resource_mode must be forwarding, interfaces, or ovs")
    _positive_int(data.get("episode_steps"), "episode_steps")
    _positive_number(data.get("decision_seconds"), "decision_seconds")
    _positive_number(data.get("warmup_seconds"), "warmup_seconds", allow_zero=True)
    for key in ("order", "eval_points"):
        _positive_int(data["embedding"].get(key), "embedding." + key)
    if data["embedding"]["normalization"] not in ("identity", "source"):
        raise ValueError("Select identity or source normalization")
    reward = data["reward"]
    for key in ("alpha", "throughput_reference_bps", "power_reference_watts"):
        _positive_number(reward.get(key), "reward." + key, allow_zero=key == "alpha")
    for key in ("idle_switch_watts", "idle_port_watts", "dynamic_watts_per_mbps"):
        _positive_number(
            reward.get("power_model", {}).get(key),
            "reward.power_model." + key,
            allow_zero=True,
        )
    for event in data["failures"]:
        _positive_int(event.get("step"), "failure.step", allow_zero=True)
        if event["step"] >= data["episode_steps"] or event.get("kind") not in (
            "link",
            "controller",
        ):
            raise ValueError(
                "Failure needs a step in the trial and kind link or controller",
            )
    return data


def training_stages(
    settings: dict[str, Any],
    episodes: int,
    traffic: str,
    known_profiles: Collection[str],
) -> list[dict[str, Any]]:
    learning = settings["learning"]
    _positive_int(learning.get("batch_size"), "learning.batch_size")
    for rate in ("actor_lr", "critic_lr", "estimator_lr"):
        _positive_number(learning.get(rate), "learning." + rate)
    pruning = learning["pruning"]
    if pruning["mode"] not in ("dense", "static", "dynamic"):
        raise ValueError("pruning.mode must be dense, static or dynamic")
    if pruning["mode"] == "dynamic":
        _positive_int(pruning.get("ramp_rounds"), "pruning.ramp_rounds")
        _positive_int(
            pruning.get("warmup_rounds", 5),
            "pruning.warmup_rounds",
            allow_zero=True,
        )
    stages = learning.get("stages") or [
        {
            "name": "training",
            "mode": "online",
            "traffic": traffic,
            "episodes": episodes,
        },
    ]
    if len({stage["name"] for stage in stages}) != len(stages):
        raise ValueError("Training stage names must be distinct")
    for stage in stages:
        _positive_int(stage.get("episodes"), str(stage["name"]) + ".episodes")
        if stage.get("traffic") not in known_profiles:
            raise ValueError(
                "Set a scenario traffic profile for stage " + str(stage["name"]),
            )
        if stage["mode"] not in ("pretrain", "frozen", "online"):
            raise ValueError("Unknown estimator mode: " + str(stage["mode"]))
    return stages


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        msg = "Set {} to an integer {}".format(name, ">= 0" if allow_zero else "> 0")
        raise ValueError(msg)


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        msg = "Set {} to a finite number {}".format(
            name,
            ">= 0" if allow_zero else "> 0",
        )
        raise ValueError(msg)
