"""Run a joint policy episode and measure delivered flows after draining traffic."""

import time
from collections.abc import Callable, Sequence
from statistics import mean

import numpy as np

from .datasets import Dataset
from .flow_records import collect_flow_records
from .joint_environment import JointEnvironment
from .run_log import RunLog
from .settings import positive


def run_trial(
    environment: JointEnvironment,
    policy: Callable[[dict], Sequence[int]],
    log: RunLog,
    *,
    episode: int,
    dataset: Dataset,
    on_transition: Callable[[dict], None] | None = None,
    progress: str | None = None,
    observations: int | None = None,
    nonzero_only: bool = False,
    measure_flows: bool = True,
    steps: int | None = None,
) -> dict:
    if steps is not None:
        positive(steps, "steps", integer=True)
    log.write(
        "trial_start",
        episode=episode,
        dataset=dataset.name,
        dataset_sha256=dataset.fingerprint,
    )
    embeddings = []
    cumulative = energy = elapsed = 0.0
    started = time.monotonic()
    success = False
    try:
        observation, initial = environment.reset(
            dataset, measure_flows=measure_flows, steps=steps
        )
        log.write(
            "reset",
            episode=episode,
            observation=observation["global_obs"].tolist(),
            **initial,
        )
        decisions = dataset.decisions(environment.settings) if steps is None else steps
        for decision in range(1, decisions + 1):
            actions = np.asarray(policy(observation), dtype=int)
            next_observation, rewards, terminated, truncated, info = environment.step(
                actions
            )
            transition = {
                "state": observation["global_obs"].tolist(),
                "action": actions.tolist(),
                "reward": rewards.tolist(),
                "next_state": next_observation["global_obs"].tolist(),
                "terminated": terminated,
                "truncated": truncated,
            }
            log.write(
                "transition", episode=episode, decision=decision, **transition, **info
            )
            average = float(np.mean(rewards))
            cumulative += average
            elapsed += info["step_seconds"]
            energy += info["step_seconds"] * sum(
                part["power_watts"] for part in info["reward_components"]
            )
            has_traffic = any(
                part["total_delivered_bps"] > 0 for part in info["reward_components"]
            )
            if not nonzero_only or has_traffic:
                embeddings.append(info["raw_embedding"])
            if on_transition:
                on_transition(transition)
            if progress:
                print(
                    f"{progress} | decision {decision}/{decisions} | "
                    f"remaining {decisions - decision} | mean reward {average:.6f}",
                    flush=True,
                )
            observation = next_observation
            enough_observations = (
                observations is not None and len(embeddings) >= observations
            )
            if terminated or truncated or enough_observations:
                break
        if observations is not None and len(embeddings) < observations:
            raise ValueError(
                f"{dataset.name} yielded {len(embeddings)} of {observations} requested observations"
            )
        success = True
    finally:
        result = environment.emulation.finish_traffic(
            environment.settings["drain_seconds"] if success else 0
        )
    complete = [
        row
        for row in collect_flow_records(result, log.capture_cache)
        if row["fct_seconds"] is not None
    ]
    for row in complete:
        log.write("flow", episode=episode, **row)
    summary = {
        "episode": episode,
        "cumulative_reward": cumulative,
        "power_watts": energy / elapsed if elapsed else None,
        "fct_seconds": mean(row["fct_seconds"] for row in complete)
        if complete
        else None,
        "completed_flows": len(complete),
        "measured_seconds": elapsed,
        "wall_seconds": time.monotonic() - started,
    }
    log.write("trial_end", **summary)
    return dict(summary, embeddings=embeddings)
