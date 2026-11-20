"""Coordinate one action and one telemetry window for every logical agent."""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .controller_client import (
    ControllerClient,
    ControllerUnavailableError,
    JointActionError,
)
from .emulation import Emulation
from .datasets import Dataset
from .observations import ObservationEncoder
from .reward import compute_reward
from .scenario import Scenario
from .settings import positive


class JointEnvironment:
    """Actions [agent] are binary and observations expose global_obs [features].

    Rewards are a float vector in scenario order. The configured episode horizon
    is a time limit, so it truncates a transition without terminating the MDP.
    """

    def __init__(
        self,
        scenario: Scenario,
        settings: dict[str, Any],
        client: ControllerClient,
        emulation: Emulation,
        encoder: ObservationEncoder,
        encode_local: Callable[[list[dict[str, Any]]], list[NDArray]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.settings = settings
        self.client = client
        self.emulation = emulation
        self.encoder = encoder
        self.encode_local = encode_local
        self.slots = scenario.controller_ids
        self.action_space = spaces.MultiDiscrete(np.full(len(self.slots), 2))
        self.observation_space = spaces.Dict(
            {
                "global_obs": spaces.Box(
                    -np.inf, np.inf, shape=(encoder.size,), dtype=np.float32
                ),
            }
        )
        self.reward_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.slots),), dtype=np.float32
        )
        self.sample_timeout = settings.get("sample_timeout_seconds", 15)
        self.steps = 0
        self.snapshots = None
        self.traffic = None

    def _wait_for_samples(
        self,
        after: float,
        expected: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.sample_timeout
        last = "no samples"
        with ThreadPoolExecutor(max_workers=len(self.slots)) as executor:
            while time.monotonic() < deadline:
                jobs = [
                    executor.submit(self.client.fetch_stats, slot)
                    for slot in self.slots
                ]
                try:
                    rows = [job.result() for job in jobs]
                except ControllerUnavailableError as error:
                    last = str(error)
                else:
                    earliest = min(row["sampled_at"] for row in rows)
                    latest = max(row["sampled_at"] for row in rows)
                    poll = self.scenario.data["telemetry"]["poll_seconds"]
                    if (
                        latest - earliest <= 3 * poll
                        and time.time() - earliest <= 3 * poll
                        and all(row["window_start"] >= after for row in rows)
                        and all(
                            row["resources"][slot]["action"] == ack["action"]
                            and row["resources"][slot]["revision"] >= ack["revision"]
                            for row in rows
                            for slot, ack in expected.items()
                        )
                    ):
                        return rows
                    last = "waiting for post-action, synchronized telemetry"
                time.sleep(0.15)
        raise ControllerUnavailableError(last)

    def _observe(self) -> tuple[dict[str, NDArray], list[float]]:
        topologies = [row["topology"] for row in self.snapshots]
        embeddings = (
            self.encode_local(topologies)
            if self.encode_local
            else [self.encoder.embed_local(topology) for topology in topologies]
        )
        raw = np.concatenate(embeddings)
        return {"global_obs": self.encoder.normalize(raw)}, raw.tolist()

    def _apply(
        self,
        actions: Sequence[int] | NDArray,
    ) -> dict[str, dict[str, Any]]:
        mapped = dict(zip(self.slots, map(int, actions), strict=True))
        acknowledgments = self.client.apply_actions(
            mapped, time.monotonic() + self.sample_timeout
        )
        return acknowledgments

    def reset(
        self,
        dataset: Dataset,
        *,
        measure_flows: bool = True,
        steps: int | None = None,
    ) -> tuple[dict[str, NDArray], dict[str, Any]]:
        """Restore forwarding and start the prepared traffic schedule."""
        self.steps = 0
        self.snapshots = None
        if steps is not None:
            positive(steps, "steps", integer=True)
        horizon = dataset.decisions(self.settings) if steps is None else steps
        self.settings = dict(self.settings, episode_steps=horizon)
        self.emulation.reset(self.scenario.fingerprint)
        actions = self._apply([1] * len(self.slots))
        after = max(ack["accepted_at"] for ack in actions.values())
        self._wait_for_samples(after, actions)
        self.traffic = self.emulation.start_traffic(
            dataset.jobs,
            dataset.duration,
            measure_flows=measure_flows,
        )
        starts = (
            self.traffic.get("started_at", time.time())
            + self.settings["warmup_seconds"]
        )
        time.sleep(max(0.0, starts - time.time()))
        self.snapshots = self._wait_for_samples(starts, actions)
        observation, raw = self._observe()
        return observation, {
            "raw_embedding": raw,
            "snapshots": self.snapshots,
            "traffic": self.traffic,
            "acks": actions,
        }

    def step(
        self,
        actions: Sequence[int] | NDArray,
    ) -> tuple[dict[str, NDArray], NDArray, bool, bool, dict[str, Any]]:
        """Submit all actions before collecting the next fresh telemetry window."""
        actions = np.asarray(actions)
        if not self.action_space.contains(actions) or not np.issubdtype(
            actions.dtype, np.integer
        ):
            raise ValueError("Expected one integer binary action per controller")
        started = time.monotonic()
        acknowledgments = self._apply(actions)
        after = max(ack["accepted_at"] for ack in acknowledgments.values())
        time.sleep(
            max(0.0, self.settings["decision_seconds"] - (time.monotonic() - started))
        )
        try:
            self.snapshots = self._wait_for_samples(after, acknowledgments)
        except ControllerUnavailableError as error:
            raise JointActionError(
                acknowledgments, {"post-action telemetry": str(error)}
            ) from error
        obs, raw = self._observe()
        parts = [compute_reward(row, self.settings["reward"]) for row in self.snapshots]
        rewards = np.asarray([part["reward"] for part in parts], dtype=np.float32)
        self.steps += 1
        return (
            obs,
            rewards,
            False,
            self.steps >= self.settings["episode_steps"],
            {
                "raw_embedding": raw,
                "snapshots": self.snapshots,
                "acks": acknowledgments,
                "reward_components": parts,
                "step_seconds": time.monotonic() - started,
            },
        )


def make_environment(
    args: argparse.Namespace,
    network: Scenario,
    settings: dict,
    encoder: ObservationEncoder,
) -> JointEnvironment:
    return JointEnvironment(
        network,
        settings,
        ControllerClient(network),
        Emulation(args.mininet_socket, args.ryu_socket),
        encoder,
    )
