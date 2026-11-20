"""Resolve prepared datasets against the hosts of a generated network."""

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from ipaddress import IPv4Interface
from pathlib import Path

from EAGLE.src.data_preparation.hosts import dataset_hosts
from EAGLE.src.network import Network
from EAGLE.src.paths import ROOT
from EAGLE.src.settings import positive

DATASET_GROUPS = {
    "sources": ("dc_t0", "mg_t0"),
    "loads": ("dc_t1", "dc_t3", "dc_t4"),
    "congestions": tuple(f"dc_t0_trenv{index}" for index in range(2, 6)),
}


@lru_cache(maxsize=512)
def _capture_digest(path: Path, _size: int, _modified: int) -> str:
    """Cache capture hashes using the path, size, and nanosecond modification time."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Dataset:
    """A prepared traffic schedule and the digest of its manifest and captures."""

    path: Path
    duration: float
    jobs: list[dict]
    fingerprint: str

    @property
    def name(self) -> str:
        return self.path.stem

    def decisions(self, settings: dict) -> int:
        count = math.floor(
            (self.duration - settings["warmup_seconds"]) / settings["decision_seconds"]
        )
        if count < 1:
            raise ValueError(f"{self.path} is shorter than one decision interval")
        return count


def load_dataset(
    network: Network, name: str, split: str, *, root: Path = ROOT
) -> Dataset:
    """Resolve a host-count dataset, check host identities, and build replay jobs.

    PCAP digests are included in the dataset fingerprint so a changed capture
    cannot silently reuse a training checkpoint or evaluation result."""
    if split not in ("train", "test") or Path(name).name != name:
        raise ValueError("Select a dataset name and train or test split")
    path = (
        root
        / "datasets"
        / f"{len(network.data['hosts'])}_hosts"
        / split
        / (name + ".json")
    )
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError(f"Expected dataset schema_version 1 in {path}")

    def host_key(rows: list[dict]) -> dict:
        return {
            row["id"]: (str(IPv4Interface(row["ip"]).ip), row["mac"].lower())
            for row in rows
        }

    if host_key(dataset_hosts(path)) != host_key(network.data["hosts"]):
        raise ValueError(
            f"Host identities or addresses in {path} differ from {network.path.name}"
        )
    duration = data["duration_seconds"]
    positive(duration, f"{path}: duration_seconds")
    jobs = data.get("jobs")
    if jobs is None:
        jobs = []
        for index, row in enumerate(data["captures"]):
            if not row["packets"]:
                continue
            jobs.append(
                {
                    "id": f"replay-{index + 1}",
                    "kind": "pcap",
                    "source": row["source"],
                    "target": row["target"],
                    "start_seconds": row.get("first_packet_seconds") or 0,
                    "path": row["path"],
                }
            )
    names = {row["id"] for row in network.data["hosts"]}
    identifiers = set()
    for job in jobs:
        if (
            job["id"] in identifiers
            or job["source"] not in names
            or job["target"] not in names
            or job["source"] == job["target"]
        ):
            raise ValueError(f"Invalid job in {path}")
        identifiers.add(job["id"])
        positive(job["start_seconds"], f"{path}: start_seconds", zero=True)
        if job["start_seconds"] >= duration:
            raise ValueError(f"Job falls outside the duration in {path}")
        if job["kind"] == "tcp":
            positive(job["size_bytes"], f"{path}: size_bytes", integer=True)
            for key in ("rate_bps", "timeout_seconds"):
                positive(job[key], f"{path}: {key}")
        elif job["kind"] == "pcap":
            capture = (path.parent / job["path"]).resolve()
            stat = capture.stat()
            job["path"] = str(capture)
            job["sha256"] = _capture_digest(capture, stat.st_size, stat.st_mtime_ns)
        else:
            raise ValueError(f"Unknown traffic kind in {path}: {job['kind']}")
    capture_hashes = "".join(job["sha256"] for job in jobs if job["kind"] == "pcap")
    return Dataset(
        path, duration, jobs, hashlib.sha256(raw + capture_hashes.encode()).hexdigest()
    )
