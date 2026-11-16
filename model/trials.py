"""Run single episodes and keep measurements for actions, rewards, and flows."""

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .controller_client import JointActionError
from .flow_records import collect_flow_records
from .joint_environment import JointEnvironment
from .reward import compute_carbon_grams
from .run_log import RunLog


def run_trial(
    environment: JointEnvironment,
    policy: Callable[
        [dict[str, NDArray], list[dict[str, Any]] | None],
        Sequence[int] | NDArray,
    ],
    log: RunLog,
    *,
    episode: int,
    traffic: str,
    stage: str | None = None,
    on_transition: Callable[[dict[str, Any]], None] | None = None,
    keep_captures: bool = False,
    external_flows: Iterable[dict[str, Any]] = (),
) -> None:
    log.write("trial_start", episode=episode, traffic_profile=traffic, stage=stage)
    success = False
    cumulative_reward = 0.0
    started_at = time.monotonic()
    try:
        observation, initial = environment.reset(traffic)
        log.write(
            "reset",
            episode=episode,
            observation=observation["global_obs"].tolist(),
            **initial,
        )
        for step in range(environment.settings["episode_steps"]):
            start = time.monotonic()
            actions = policy(observation, environment.snapshots)
            inference = time.monotonic() - start
            next_observation, rewards, terminated, truncated, info = environment.step(
                actions,
            )
            transition = {
                "state": observation["global_obs"].tolist(),
                "action": np.asarray(actions, dtype=int).tolist(),
                "reward": rewards.tolist(),
                "next_state": next_observation["global_obs"].tolist(),
                "terminated": terminated,
                "truncated": truncated,
            }
            cumulative_reward += float(rewards.mean())
            carbon = _carbon_sample(environment.settings, info)
            log.write(
                "transition",
                episode=episode,
                step=step,
                inference_seconds=inference,
                carbon_grams=carbon,
                **transition,
                **info,
            )
            if on_transition is not None:
                on_transition(transition)
            observation = next_observation
            if terminated or truncated:
                break
        success = True
    except BaseException as error:
        detail = {}
        if isinstance(error, JointActionError):
            detail = {"acks": error.acknowledgments, "failures": error.failures}
        log.write("trial_error", episode=episode, message=str(error), **detail)
        raise
    finally:
        try:
            result = environment.emulation.finish_traffic()
            log.write("traffic_finish", episode=episode, **result)
            if keep_captures:
                log.archive_captures(episode, result)
            for row in collect_flow_records(result, log.capture_cache):
                log.write("flow", episode=episode, **row)
            for row in external_flows:
                log.write("flow", episode=episode, **row)
        except Exception as error:
            log.write(
                "trial_error",
                episode=episode,
                message="Traffic evidence: " + str(error),
            )
            if success:
                raise
    log.write(
        "trial_end",
        episode=episode,
        cumulative_reward=cumulative_reward,
        wall_seconds=time.monotonic() - started_at,
    )


def _carbon_sample(settings: dict[str, Any], info: dict[str, Any]) -> float | None:
    config = settings.get("carbon")
    if config is None:
        return None
    power = settings["reward"]["power_model"]
    links = set()
    egress = {}
    for snapshot in info["snapshots"]:
        links.update(tuple(edge) for edge in snapshot["usage"]["links"])
        for node, rate in snapshot["usage"]["switch_egress_bps"].items():
            egress[node] = rate / 1_000_000 * power["dynamic_watts_per_mbps"]
    return compute_carbon_grams(
        info["step_seconds"],
        egress,
        links,
        config["intensity_grams_per_kwh"],
        power["idle_port_watts"],
    )
