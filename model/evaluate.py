"""Run frozen checkpoint, fixed-action, or external policy trials."""
# ruff: noqa: E731

import argparse
import hashlib
import random
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import load_inference_model
from .cli import add_trial_arguments, parse_trial_inputs, resolve_device
from .controller_client import ControllerClient
from .emulation import Emulation
from .joint_environment import JointEnvironment
from .observations import make_encoder
from .run_log import RunLog, read_calibration, read_events
from .scenario import Scenario
from .trials import run_trial


def evaluate(
    args: argparse.Namespace,
    scenario: Scenario,
    settings: dict[str, Any],
) -> None:
    np.random.seed(args.seed % (1 << 32))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    slots = scenario.controller_ids
    calibration = (
        read_calibration(args.normalize_from)
        if args.normalize_from and not args.checkpoint
        else ()
    )
    device = None
    if args.checkpoint:
        device = resolve_device(args.device)
        policy, encoder = load_inference_model(args.checkpoint, scenario)
        policy.to(device)
        decide = lambda observation, _: policy.decide(observation)
        method = args.method or "EAGLE"
    elif args.fixed:
        actions = [int(item) for item in args.fixed.split(",")]
        if len(actions) != len(slots) or any(item not in (0, 1) for item in actions):
            raise ValueError("Use one comma-separated binary action per controller")
        encoder = make_encoder(scenario, settings, calibration)
        decide = lambda *_: actions
        method = args.method or "fixed"
    else:
        module, separator, function = args.policy.partition(":")
        if not separator:
            raise ValueError("Use --policy MODULE:FUNCTION")
        external = getattr(import_module(module), function)
        encoder = make_encoder(scenario, settings, calibration)
        decide = external
        method = args.method or args.policy

    hashes = {}
    if args.checkpoint:
        with args.checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        hashes[str(args.checkpoint)] = digest
    log = RunLog(
        args.output,
        scenario,
        settings,
        method=method,
        condition=args.condition or scenario.data["id"],
        seed=args.seed,
        expected_episodes=args.episodes,
        encoder=encoder.to_dict(),
        traffic_profile=args.traffic,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        fixed_action=args.fixed,
        external_policy=args.policy,
        device=str(device) if device is not None else None,
        checkpoint_sha256=hashes,
        flow_records_path=str(args.flow_records) if args.flow_records else None,
        calibration=(
            str(args.normalize_from)
            if args.normalize_from and not args.checkpoint
            else None
        ),
    )
    environment = JointEnvironment(
        scenario,
        settings,
        ControllerClient(scenario),
        Emulation(
            args.mininet_socket, args.ryu_socket, discover=args.traffic != "external"
        ),
        encoder,
    )
    external_flows = {}
    if args.flow_records:
        for row in read_events(args.flow_records):
            row.pop("event", None)
            external_flows.setdefault(row.pop("episode", 1), []).append(row)
    for episode in range(1, args.episodes + 1):
        run_trial(
            environment,
            decide,
            log,
            episode=episode,
            traffic=args.traffic,
            keep_captures=args.keep_captures,
            external_flows=external_flows.get(episode, ()),
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_trial_arguments(parser)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint", type=Path)
    modes.add_argument("--fixed", metavar="0,1,0")
    modes.add_argument(
        "--policy",
        metavar="MODULE:FUNCTION",
        help="Call an external baseline with observation and controller snapshots",
    )
    parser.add_argument("--method", help="Name to use in comparative reports")
    parser.add_argument("--condition", help="Name shared by matched baseline trials")
    parser.add_argument(
        "--identity-normalization",
        action="store_true",
        help="Save raw embeddings before fitting source normalization",
    )
    parser.add_argument(
        "--normalize-from",
        type=Path,
        help="Source calibration events.jsonl for external policies",
    )
    parser.add_argument(
        "--flow-records", type=Path, help="External flow measurements as JSON lines"
    )
    args = parser.parse_args()
    scenario, settings = parse_trial_inputs(
        parser, args, normalization="identity" if args.identity_normalization else None
    )
    try:
        evaluate(args, scenario, settings)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    _main()
