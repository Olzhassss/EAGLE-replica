"""Train EAGLE over a scenario curriculum and save resumable CPU checkpoints."""
# ruff: noqa: B023

import argparse
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint, save_checkpoint
from .cli import add_trial_arguments, parse_trial_inputs, resolve_device
from .controller_client import ControllerClient
from .emulation import Emulation
from .joint_environment import JointEnvironment
from .observations import ObservationEncoder, make_encoder
from .run_log import RunLog, read_calibration, read_events
from .scenario import Scenario
from .settings import training_stages
from .trials import run_trial
from .workers import AgentPool


def _batches(
    rows: Iterable[dict[str, Any]],
    size: int,
) -> Iterator[list[dict[str, Any]]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _round_batch(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = ("state", "action", "reward", "next_state", "terminated")
    return {field: [row[field] for row in rows] for field in fields}


def train(
    args: argparse.Namespace,
    scenario: Scenario,
    settings: dict[str, Any],
    stages: list[dict[str, Any]],
    encoder: ObservationEncoder,
    resume: dict[str, Any] | None,
) -> None:
    device = resolve_device(args.device)
    log = RunLog(
        args.output,
        scenario,
        settings,
        method="EAGLE-training",
        seed=args.seed,
        encoder=encoder.to_dict(),
        stages=stages,
        worker_mode=args.workers,
        device=str(device),
        resume=str(args.resume) if args.resume else None,
        calibration=str(args.normalize_from) if args.normalize_from else None,
        replay=str(args.replay_from) if args.replay_from else None,
        condition=scenario.data["id"],
        expected_episodes=sum(item["episodes"] for item in stages),
    )
    print(f"Training device: {device}", flush=True)
    round_number = resume["round"] if resume else 0
    episode = 0
    with AgentPool(
        encoder,
        settings["learning"],
        args.seed,
        distributed=args.workers == "process",
        device=device,
    ) as agents:
        if resume:
            agents.load_state_dict(resume["agents"])
        environment = JointEnvironment(
            scenario,
            settings,
            ControllerClient(scenario),
            Emulation(
                args.mininet_socket,
                args.ryu_socket,
                discover=not args.replay_from
                and any(stage["traffic"] != "external" for stage in stages),
            ),
            encoder,
            encode_local=agents.embed,
        )
        for stage in stages:
            recorded = (
                _offline_rows(args.replay_from, stage["traffic"], encoder)
                if args.replay_from
                else None
            )
            for _ in range(stage["episodes"]):
                episode += 1
                if resume and episode <= resume["episode"]:
                    continue
                batch = []

                def update() -> None:
                    nonlocal round_number
                    round_number += 1
                    before = agents.bytes_transferred
                    started = time.monotonic()
                    result = agents.run_round(
                        round_number,
                        stage["mode"],
                        _round_batch(batch),
                        stage.get("active_agents"),
                    )
                    log.write(
                        "learning",
                        episode=episode,
                        stage=stage["name"],
                        batch_size=len(batch),
                        elapsed_seconds=time.monotonic() - started,
                        worker_transport_bytes=agents.bytes_transferred - before,
                        **result,
                    )
                    batch.clear()

                def collect(transition: dict[str, Any]) -> None:
                    batch.append(transition)
                    if len(batch) == settings["learning"]["batch_size"]:
                        update()

                if recorded is not None:
                    log.write(
                        "trial_start",
                        episode=episode,
                        traffic_profile=stage["traffic"],
                        stage=stage["name"],
                    )
                    for transitions in _batches(recorded, settings["episode_steps"]):
                        for row in transitions:
                            collect(row)
                        break
                    else:
                        raise ValueError(
                            "Replay has no remaining transitions for "
                            + stage["traffic"],
                        )
                    log.write("trial_end", episode=episode, offline=True)
                else:
                    run_trial(
                        environment,
                        lambda obs, _: agents.decide(obs),
                        log,
                        episode=episode,
                        traffic=stage["traffic"],
                        stage=stage["name"],
                        on_transition=collect,
                        keep_captures=args.keep_captures,
                    )
                if batch:
                    update()
                save_checkpoint(
                    args.output / "checkpoint.pth",
                    {
                        "encoder": encoder.to_dict(),
                        "agents": agents.state_dict(),
                        "seed": args.seed,
                        "round": round_number,
                        "episode": episode,
                        "stage": stage["name"],
                        "scenario_sha256": scenario.fingerprint,
                        "settings": settings,
                        "stages": stages,
                    },
                )
                print(
                    "Saved checkpoint for episode {} ({})".format(
                        episode,
                        stage["name"],
                    ),
                    flush=True,
                )


def _offline_rows(
    file: Path,
    traffic: str,
    encoder: ObservationEncoder,
) -> Iterator[dict[str, Any]]:
    current = None
    previous = None
    for row in read_events(file):
        if row["event"] == "trial_start":
            current = row["traffic_profile"]
            previous = None
        elif row["event"] == "reset" and current == traffic:
            previous = encoder.normalize(row["raw_embedding"]).tolist()
        elif row["event"] == "transition" and current == traffic:
            if previous is None:
                raise ValueError(
                    "Replay transition is missing its preceding observation",
                )
            observation = encoder.normalize(row["raw_embedding"]).tolist()
            yield dict(row, state=previous, next_state=observation)
            previous = observation


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_trial_arguments(parser)
    parser.add_argument("--workers", choices=("process", "local"), default="process")
    parser.add_argument(
        "--normalize-from",
        type=Path,
        help="Source run events.jsonl with raw graph embeddings",
    )
    parser.add_argument(
        "--replay-from",
        type=Path,
        help="Train from saved transitions without running Mininet or Ryu",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    scenario, settings = parse_trial_inputs(parser, args)
    traffic_profiles = set(scenario.data.get("traffic", {})) | {"idle", "external"}
    try:
        stages = training_stages(
            settings,
            args.episodes,
            args.traffic,
            traffic_profiles,
        )
        resume = load_checkpoint(args.resume) if args.resume else None
        if resume:
            if (
                resume["seed"] != args.seed
                or resume["scenario_sha256"] != scenario.fingerprint
                or resume["settings"] != settings
                or resume["stages"] != stages
            ):
                raise ValueError(
                    "Resume requires the original scenario, settings, schedule, and seed",
                )
            if args.replay_from:
                raise ValueError(
                    "Offline replay resumes require a new explicit input window",
                )
            encoder = ObservationEncoder.from_dict(resume["encoder"])
        else:
            calibration = (
                read_calibration(args.normalize_from) if args.normalize_from else ()
            )
            encoder = make_encoder(scenario, settings, calibration)
        train(args, scenario, settings, stages, encoder, resume)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    _main()
