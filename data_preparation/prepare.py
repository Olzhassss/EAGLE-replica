"""Prepare Digital Corpora and Megalodon datasets for Mininet experiments."""

import argparse
import json
import math
from itertools import permutations
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from config import Config

from .analyze_trace import DC_CAPTURE, MG_CAPTURE, ROOT
from .load_and_split import Window, load_windows, pool_windows, split_dc
from .replay import generate_pcap_dataset, hosts_for
from .traffic_congestions import generate_congestions
from .traffic_loads import generate_loads, generate_synthetic_dataset


def prepare_datasets(
    dc: Path,
    mg: Path,
    output: Path,
    hosts: int,
    min_decisions: int,
    seed: int,
    decision_seconds: float = Config.DECISION_SECONDS,
    *,
    skip_pcaps: bool = False,
    skip_train: bool = False,
) -> None:
    if type(hosts) is not int or hosts < 2:
        raise ValueError("At least two hosts are required")
    if (
        type(min_decisions) is not int
        or min_decisions < 1
        or not math.isfinite(decision_seconds)
        or decision_seconds <= 0
    ):
        raise ValueError("min_decisions and decision_seconds must be positive")
    time_scale = Config.TRAFFIC_SCALE_FACTOR
    if time_scale is None:
        time_scale = hosts
    if (
        type(time_scale) not in (int, float)
        or not math.isfinite(time_scale)
        or time_scale < 1
    ):
        raise ValueError(
            "TRAFFIC_SCALE_FACTOR must be at least 1, or None to use the host count"
        )
    pairs = list(permutations(hosts_for(hosts, None), 2))
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        splits = ("test",) if skip_train else ("train", "test")
        kinds = ("t0",) if skip_pcaps else ("pcap", "t0")
        for split in splits:
            (staging / split).mkdir(parents=True)

        def write(split: str, name: str, windows: list[Window]) -> None:
            seconds = len(windows) // len(pairs)
            if math.floor(seconds * time_scale / decision_seconds) < min_decisions:
                seconds = math.ceil(min_decisions * decision_seconds / time_scale)
                # Account for floating-point rounding in the reported decision count.
                if math.floor(seconds * time_scale / decision_seconds) < min_decisions:
                    seconds += 1
            pools = pool_windows(
                windows,
                len(pairs),
                seconds,
                Random(f"{seed}:{split}:{name}"),
            )
            if not skip_pcaps:
                generate_pcap_dataset(
                    staging / split / f"{name}_pcap.json",
                    pools,
                    pairs,
                    time_scale=time_scale,
                )
            generate_synthetic_dataset(
                staging / split / f"{name}_t0.json",
                pools,
                pairs,
                time_scale=time_scale,
            )
            for kind in kinds:
                path = staging / split / f"{name}_{kind}.json"
                data = json.loads(path.read_text())
                data.update(
                    decision_seconds=decision_seconds,
                    decisions=math.floor(data["duration_seconds"] / decision_seconds),
                    seed=seed,
                )
                path.write_text(json.dumps(data, indent=2) + "\n")

        with TemporaryDirectory(dir=temporary) as spool:
            train, test = split_dc(
                load_windows(dc, Path(spool) / "dc", store_packets=not skip_pcaps),
                Random(seed),
            )
            if not skip_train:
                write("train", "dc", train)
            write("test", "dc", test)
        with TemporaryDirectory(dir=temporary) as spool:
            write(
                "test",
                "mg",
                load_windows(
                    mg,
                    Path(spool) / "mg",
                    megalodon=True,
                    store_packets=not skip_pcaps,
                ),
            )
        generate_loads(staging / "test")
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    base = commands.add_parser(
        "default", help="Generate base datasets and global load variants"
    )
    congestion = commands.add_parser(
        "congestion", help="Generate regional variants from DC T0"
    )
    for command in (base, congestion):
        command.add_argument("--hosts", type=int, required=True)
        command.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
        command.add_argument(
            "--skip-train", action="store_true", help="Generate test datasets only"
        )
    base.add_argument(
        "--dc", type=Path, default=DC_CAPTURE, help="Digital Corpora capture"
    )
    base.add_argument(
        "--mg", type=Path, default=MG_CAPTURE, help="Megalodon Content Server capture"
    )
    base.add_argument("--min-decisions", type=int, default=Config.MIN_DECISIONS)
    base.add_argument("--decision-seconds", type=float, default=Config.DECISION_SECONDS)
    base.add_argument("--seed", type=int, default=Config.SEED)
    base.add_argument(
        "--skip-pcaps",
        action="store_true",
        help="Generate synthetic datasets without spooling packets",
    )
    congestion.add_argument(
        "--network", type=Path, required=True, help="Partitioned scenario JSON"
    )
    args = parser.parse_args()
    if args.hosts < 2:
        parser.error("--hosts must be at least 2")
    output = args.output_dir / f"{args.hosts}_hosts"
    try:
        if args.command == "default":
            prepare_datasets(
                args.dc,
                args.mg,
                output,
                args.hosts,
                args.min_decisions,
                args.seed,
                args.decision_seconds,
                skip_pcaps=args.skip_pcaps,
                skip_train=args.skip_train,
            )
        else:
            generate_congestions(
                args.network, output, args.hosts, skip_train=args.skip_train
            )
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
