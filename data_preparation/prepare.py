"""Prepare Digital Corpora and Megalodon datasets for Mininet experiments."""

import argparse
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

MIN_STEPS = 12


def prepare_datasets(
    dc: Path,
    mg: Path,
    output: Path,
    hosts: int,
    steps: int,
    seed: int,
    scenario: Path | None,
) -> None:
    if hosts < 2:
        raise ValueError("At least two hosts are required")
    if steps < MIN_STEPS:
        raise ValueError(f"--episode-steps must be at least {MIN_STEPS}")
    pairs = list(permutations(hosts_for(hosts, scenario), 2))
    rng = Random(seed)
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        for split in ("train", "test"):
            (staging / split).mkdir(parents=True)

        def write(split: str, name: str, windows: list[Window]) -> None:
            pools = pool_windows(windows, len(pairs), steps, rng)
            generate_pcap_dataset(staging / split / f"{name}_pcap.json", pools, pairs)
            generate_synthetic_dataset(staging / split / f"{name}_t0.json", pools, pairs)

        with TemporaryDirectory(dir=temporary) as spool:
            train, test = split_dc(load_windows(dc, Path(spool) / "dc"), rng)
            write("train", "dc", train)
            write("test", "dc", test)
        with TemporaryDirectory(dir=temporary) as spool:
            write("test", "mg", load_windows(mg, Path(spool) / "mg", megalodon=True))
        generate_loads(staging / "test")
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    base = commands.add_parser("default", help="Generate base datasets and global load variants")
    congestion = commands.add_parser("congestion", help="Generate regional variants from DC T0")
    for command in (base, congestion):
        command.add_argument("--hosts", type=int, required=True)
        command.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    base.add_argument("--dc", type=Path, default=DC_CAPTURE, help="Digital Corpora capture")
    base.add_argument(
        "--mg", type=Path, default=MG_CAPTURE, help="Megalodon Content Server capture"
    )
    base.add_argument("--episode-steps", type=int, default=MIN_STEPS)
    base.add_argument("--seed", type=int, default=Config.SEED)
    base.add_argument("--scenario", type=Path, help="Use these host IDs, MACs and IPv4 addresses")
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
                args.dc, args.mg, output, args.hosts, args.episode_steps, args.seed, args.scenario
            )
        else:
            generate_congestions(args.network, output, args.hosts)
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
