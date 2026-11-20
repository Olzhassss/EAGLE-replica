"""Generate the three single-community hotspots and an all-host hotspot."""

import json
from fcntl import LOCK_EX, flock
from pathlib import Path
from tempfile import TemporaryDirectory

from EAGLE.src.data_preparation.synthetic import scale_synthetic

SCALE_FACTOR = 1.5


def _communities(network: dict) -> list[set[str]]:
    """Resolve each host attachment to its controller domain for regional scaling."""
    communities = {controller["id"]: set() for controller in network["controllers"]}
    if len(communities) != 3 or len(network["controllers"]) != 3:
        raise ValueError(
            "Regional congestion requires exactly three distinct controllers"
        )
    switches = {switch["id"]: switch["controller"] for switch in network["switches"]}
    hosts = {host["id"] for host in network["hosts"]}
    if len(hosts) != len(network["hosts"]):
        raise ValueError("Network host IDs must be distinct")
    attachments = {}
    for link in network["links"]:
        source, target = link["source"], link["target"]
        if target in hosts:
            source, target = target, source
        if source in hosts:
            if source in attachments or target not in switches:
                raise ValueError(f"Expected one access switch for {source}")
            attachments[source] = target
    if attachments.keys() != hosts:
        raise ValueError("Network must attach every host to an access switch")
    for host, switch in attachments.items():
        communities[switches[switch]].add(host)
    return list(communities.values())


def generate_congestions(
    network: Path,
    dataset: Path,
    hosts: int,
    *,
    skip_train: bool = False,
) -> None:
    """Stage per-domain and global hotspots, then replace complete dataset variants."""
    communities = _communities(json.loads(network.read_text()))
    all_hosts = set().union(*communities)
    if len(all_hosts) != hosts:
        raise ValueError(f"{network} must define exactly {hosts} hosts")
    splits = ("test",) if skip_train else ("train", "test")

    with TemporaryDirectory(prefix=".congestion-", dir=dataset) as temporary:
        staging = Path(temporary)
        for split in splits:
            source = dataset / split / "dc_t0.json"
            series = json.loads(source.read_text())["series"]
            directions = {(row["source"], row["target"]) for row in series}
            if directions != {(a, b) for a in all_hosts for b in all_hosts if a != b}:
                raise ValueError(f"Dataset host directions do not match {network}")
            target = staging / split
            target.mkdir()
            for setup, sources in enumerate([*communities, all_hosts], start=2):
                scale_synthetic(
                    source, target / f"dc_t0_trenv{setup}.json", SCALE_FACTOR, sources
                )
        with (dataset / "test" / "dc_t0.json").open("rb") as baseline:
            flock(baseline, LOCK_EX)
            for split in splits:
                for path in (staging / split).iterdir():
                    path.replace(dataset / split / path.name)
