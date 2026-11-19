"""Run a frozen policy against traffic started manually in the Mininet CLI."""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .actor_critic import ActorCritic
from .artifacts import MODELS
from .checkpoints import load_inference_model
from .cli import artifact_name, load_network, resolve_device
from .controller_client import ControllerClient, ControllerUnavailableError
from .emulation import Emulation, unix_command
from .observations import ObservationEncoder
from .scenario import Scenario
from .settings import DEFAULT_SETTINGS, load_settings, positive


class InteractiveSimulation:
    """Keep HTTP/inference work off the one-second display loop."""

    def __init__(
        self,
        network: Scenario,
        policy: ActorCritic,
        encoder: ObservationEncoder,
        client: ControllerClient,
        decision_seconds: float,
        sample_timeout: float,
    ) -> None:
        self.network = network
        self.policy = policy
        self.encoder = encoder
        self.client = client
        self.decision_seconds = decision_seconds
        self.sample_timeout = sample_timeout
        self.poll_seconds = network.data["telemetry"]["poll_seconds"]
        self.stop = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.samples: dict[str, dict[str, Any]] = {}
        self.unavailable = set(network.controller_ids)
        self.message = "waiting for controller telemetry"
        self.decisions = 0
        self.error: Exception | None = None

    def _ready(
        self,
        rows: list[dict[str, Any]],
        expected: dict[str, dict[str, Any]],
        after: float,
    ) -> bool:
        if len(rows) != len(self.network.controller_ids):
            return False
        earliest = min(row["sampled_at"] for row in rows)
        latest = max(row["sampled_at"] for row in rows)
        resources = rows[0]["resources"]
        return (
            latest - earliest <= 3 * self.poll_seconds
            and time.time() - earliest <= 3 * self.poll_seconds
            and all(row["window_start"] >= after for row in rows)
            and all(row["resources"] == resources for row in rows)
            and all(
                resources[slot]["action"] == ack["action"]
                and resources[slot]["revision"] >= ack["revision"]
                for slot, ack in expected.items()
            )
        )

    def _control(self) -> None:
        slots = self.network.controller_ids
        expected, after, next_decision = {}, 0, 0
        waiting_since = time.monotonic()
        interval = min(1.0, self.poll_seconds)
        with ThreadPoolExecutor(max_workers=len(slots)) as executor:
            while not self.stop.is_set():
                started = time.monotonic()
                jobs = {
                    slot: executor.submit(self.client.fetch_stats, slot)
                    for slot in slots
                }
                samples, unavailable = {}, set()
                for slot, job in jobs.items():
                    try:
                        samples[slot] = job.result()
                    except ControllerUnavailableError:
                        unavailable.add(slot)
                rows = [samples[slot] for slot in slots if slot in samples]
                ready = self._ready(rows, expected, after)
                with self.lock:
                    self.samples.update(samples)
                    self.unavailable = unavailable
                    self.message = (
                        "" if ready else "waiting for fresh, synchronized telemetry"
                    )
                now = time.monotonic()
                if not ready:
                    if now - waiting_since >= self.sample_timeout:
                        raise ControllerUnavailableError(
                            "Timed out waiting for fresh, synchronized controller telemetry"
                        )
                else:
                    waiting_since = now
                    if now >= next_decision and not self.stop.is_set():
                        raw = np.concatenate(
                            [self.encoder.embed_local(row["topology"]) for row in rows]
                        )
                        observation = {"global_obs": self.encoder.normalize(raw)}
                        actions = np.asarray(self.policy.decide(observation))
                        if (
                            actions.shape != (len(slots),)
                            or not np.issubdtype(actions.dtype, np.integer)
                            or not np.isin(actions, [0, 1]).all()
                        ):
                            raise ValueError(
                                "Expected one integer binary action per controller"
                            )
                        with self.lock:
                            self.message = "applying policy actions"
                        expected = self.client.apply_actions(
                            dict(zip(slots, map(int, actions), strict=True)),
                            time.monotonic() + self.sample_timeout,
                        )
                        after = max(ack["accepted_at"] for ack in expected.values())
                        next_decision = now + self.decision_seconds
                        waiting_since = time.monotonic()
                        with self.lock:
                            self.decisions += 1
                            self.message = "waiting for post-action telemetry"
                self.stop.wait(max(0, interval - (time.monotonic() - started)))

    def run(self) -> None:
        try:
            self._control()
        except Exception as error:
            self.error = error
        finally:
            self.done.set()

    def status(self) -> str:
        """Show observed forwarding state; distinguish missing and stale samples."""
        with self.lock:
            samples = dict(self.samples)
            unavailable = set(self.unavailable)
            message, decisions = self.message, self.decisions
        fields = [time.strftime("%H:%M:%S")]
        for controller in self.network.data["controllers"]:
            slot, node = controller["id"], controller["support_switch"]
            sample = samples.get(slot)
            label = f"{node} ({slot})"
            if sample is None:
                fields.append(f"{label}=WAITING")
                continue
            age = max(0, time.time() - sample["sampled_at"])
            enabled = sample["resources"][slot]["action"]
            present = any(row["id"] == node for row in sample["topology"]["nodes"])
            state = (
                "ON"
                if enabled and present
                else "OFF"
                if not enabled and not present
                else "PENDING OFF"
                if not enabled
                else "UNAVAILABLE"
            )
            if slot in unavailable or age > 3 * self.poll_seconds:
                state = (
                    f"{'WAITING' if slot in unavailable else 'STALE'} (last {state})"
                )
            rate = sample["usage"]["switch_egress_bps"].get(node, 0) / 1_000_000
            fields.append(f"{label}={state} [{rate:.1f} Mb/s, age {age:.1f}s]")
        fields.append(f"decisions={decisions}")
        if message:
            fields.append(message)
        return " | ".join(fields)


def run_interactive(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    decision_seconds = (
        settings["decision_seconds"]
        if args.decision_seconds is None
        else args.decision_seconds
    )
    positive(decision_seconds, "decision_seconds")
    network = load_network(args.network)
    policy, encoder = load_inference_model(MODELS / args.model, network)
    policy.to(resolve_device(args.device))
    emulation = Emulation(args.mininet_socket)
    status = unix_command(emulation.mininet_socket, "status")
    if status["scenario"] != network.fingerprint:
        raise ValueError("Mininet network differs from the selected network file")
    simulation = InteractiveSimulation(
        network,
        policy,
        encoder,
        ControllerClient(network),
        decision_seconds,
        settings["sample_timeout_seconds"],
    )
    print(
        f"{args.network} | model={args.model} | decisions every {decision_seconds:g}s | ",
        flush=True,
    )
    worker = threading.Thread(target=simulation.run, name="interactive-policy")
    worker.start()
    try:
        next_display = time.monotonic()
        while not simulation.done.is_set():
            print(simulation.status(), flush=True)
            next_display += 1
            simulation.done.wait(max(0, next_display - time.monotonic()))
    except KeyboardInterrupt:
        print(
            "\nStopping policy; the last switch actions remain in effect.", flush=True
        )
    finally:
        simulation.stop.set()
        worker.join()
    if simulation.error is not None:
        raise simulation.error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", default="v0.json", help="Filename in networks/ (default v0.json)"
    )
    parser.add_argument(
        "--model",
        type=artifact_name,
        required=True,
        help="Inference filename in models/",
    )
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--mininet-socket", type=Path)
    parser.add_argument(
        "--decision-seconds",
        type=float,
        help="Policy cadence (default from settings); waits for post-action telemetry",
    )
    args = parser.parse_args()
    try:
        run_interactive(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
