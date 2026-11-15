"""Extract one-way TCP traffic from an Ethernet PCAP or PCAP.GZ for Mininet."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import socket
import struct
from contextlib import ExitStack
from ipaddress import IPv4Address
from math import isfinite
from pathlib import Path
from typing import Any

from .settings import DEFAULT_SCENARIO, ROOT, load_scenario

DEFAULT_RECIPE = ROOT / "scenarios/corpora-example.json"
PCAP_BYTE_ORDER = {
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">",
    b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
}


def _number(value: object, label: str, minimum: int = 0) -> int | float:
    if type(value) not in (int, float) or not isfinite(value) or value < minimum:
        raise ValueError("Invalid " + label)
    return value


def _flow_key(flow: dict[str, Any]) -> tuple[bytes, bytes | None]:
    addresses = [IPv4Address(flow[key]).packed for key in ("source_ip", "target_ip")]
    port_fields = ("source_port", "target_port")
    if not any(field in flow for field in port_fields):
        return b"".join(addresses), None
    if not all(field in flow for field in port_fields):
        raise ValueError(
            "Specify both TCP ports or omit both to select the address pair"
        )
    ports = [flow[field] for field in port_fields]
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError("TCP ports must be integers in 1..65535")
    return b"".join(addresses), struct.pack("!HH", *ports)


def _tcp_payload(frame: bytes, key: tuple[bytes, bytes | None]) -> int:
    """Return payload size for a matching untagged Ethernet IPv4 TCP direction."""
    if (
        len(frame) < 54
        or frame[12:14] != b"\x08\x00"
        or frame[23] != socket.IPPROTO_TCP
    ):
        return 0
    if frame[26:34] != key[0] or int.from_bytes(frame[20:22], "big") & 0x3FFF:
        return 0
    ip_header = (frame[14] & 15) * 4
    offset = 14 + ip_header
    if (
        ip_header < 20
        or len(frame) < offset + 20
        or (key[1] is not None and frame[offset : offset + 4] != key[1])
    ):
        return 0
    ip_total = int.from_bytes(frame[16:18], "big")
    tcp_header = (frame[offset + 12] >> 4) * 4
    payload = ip_total - ip_header - tcp_header
    if tcp_header < 20 or payload <= 0:
        return 0
    if len(frame) < 14 + ip_total:
        raise ValueError("Matching packet has truncated TCP payload")
    return payload


def _plan_jobs(
    recipe: dict[str, Any],
    scenario: dict[str, Any],
) -> list[tuple[str, dict[str, Any], bytes]]:
    if recipe.get("schema_version") != 1:
        raise ValueError("Expected recipe schema version 1")
    hosts = {
        row["id"]: bytes.fromhex(row["mac"].replace(":", ""))
        for row in scenario["hosts"]
    }
    identifiers = set()
    jobs = []
    for profile, entries in recipe["profiles"].items():
        if profile in scenario.get("traffic", {}):
            raise ValueError("Scenario already has traffic profile " + profile)
        pairs = set()
        if not entries:
            raise ValueError("Expected at least one capture job in " + profile)
        for entry in entries:
            name = entry["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
                raise ValueError("Invalid or duplicate job ID: " + name)
            identifiers.add(name)
            source, target = entry["source"], entry["target"]
            if (
                source == target
                or {source, target} - hosts.keys()
                or (source, target) in pairs
            ):
                raise ValueError(
                    "Use distinct scenario hosts and one job per ordered host pair per profile"
                )
            pairs.add((source, target))
            first, last = entry["first_frame"], entry["last_frame"]
            if (
                type(first) is not int
                or type(last) is not int
                or not 1 <= first <= last
            ):
                raise ValueError(
                    "Expected an inclusive, 1-based frame range for " + name
                )
            _number(entry["start_seconds"], "start_seconds")
            _number(entry["rate_bps"], "rate_bps", minimum=1)
            jobs.append((profile, entry, hosts[target] + hosts[source]))
    if not jobs:
        raise ValueError("Recipe has no jobs")
    return jobs


def _extract(
    capture: Path,
    key: tuple[bytes, bytes | None],
    jobs: list[tuple[str, dict[str, Any], bytes]],
    directory: Path,
) -> list[dict[str, int]]:
    pending = iter(
        sorted(range(len(jobs)), key=lambda index: jobs[index][1]["first_frame"])
    )
    next_job = next(pending, None)
    active = {}
    counts = [dict(frames=0, frame_bytes=0, payload_bytes=0) for _ in jobs]
    last = max(job["last_frame"] for _, job, _ in jobs)
    opener = gzip.open if capture.suffix.lower() == ".gz" else Path.open
    with opener(capture, "rb") as source, ExitStack() as files:
        header = source.read(24)
        endian = PCAP_BYTE_ORDER.get(header[:4])
        if (
            len(header) != 24
            or endian is None
            or struct.unpack_from(endian + "I", header, 20)[0] != 1
        ):
            raise ValueError(
                "Expected an Ethernet libpcap capture, optionally gzip-compressed"
            )
        record = struct.Struct(endian + "IIII")
        for number in range(1, last + 1):
            data = source.read(record.size)
            if len(data) != record.size:
                raise ValueError(f"Capture ended before requested frame {number}")
            _, _, captured, original = record.unpack(data)
            if not 0 < captured <= min(original, 65535):
                raise ValueError(f"Invalid Ethernet frame length at frame {number}")
            frame = source.read(captured)
            if len(frame) != captured:
                raise ValueError(f"Capture ends within frame {number}")
            while next_job is not None and jobs[next_job][1]["first_frame"] == number:
                job = jobs[next_job][1]
                stream = files.enter_context(
                    (directory / (job["id"] + ".pcap")).open("xb")
                )
                stream.write(header)
                active[next_job] = stream
                next_job = next(pending, None)
            if active:
                payload = _tcp_payload(frame, key)
                for index, stream in list(active.items()):
                    _, job, macs = jobs[index]
                    if payload:
                        if captured != original:
                            raise ValueError(f"Selected frame {number} is truncated")
                        stream.write(data)
                        stream.write(macs + frame[12:])
                        counts[index]["frames"] += 1
                        counts[index]["frame_bytes"] += original
                        counts[index]["payload_bytes"] += payload
                    if number == job["last_frame"]:
                        stream.close()
                        del active[index]
    return counts


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(
    capture: Path, scenario_path: Path, recipe_path: Path, output: Path
) -> Path:
    """Write prepared PCAPs and a scenario with relocatable replay paths."""
    capture = capture.resolve()
    scenario = load_scenario(scenario_path)
    recipe = json.loads(recipe_path.read_text())
    jobs = _plan_jobs(recipe, scenario.data)
    key = _flow_key(recipe["flow"])
    if not capture.is_file():
        raise FileNotFoundError(capture)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scenario_file = output / "scenario.json"
    try:
        counts = _extract(capture, key, jobs, output)
        capture_sha256 = _sha256(capture)
        traffic = scenario.data.setdefault("traffic", {})
        for entries in traffic.values():
            for job in entries:
                if job["kind"] == "pcap":
                    job["path"] = os.path.relpath(
                        scenario.path.parent / job["path"], output
                    )
        for (profile, job, _), count in zip(jobs, counts, strict=True):
            name = job["id"]
            if not count["frames"]:
                raise ValueError("No matching TCP payload frames for " + name)
            if count["payload_bytes"] >= 1 << 32:
                raise ValueError(
                    "Split "
                    + name
                    + " into shorter windows to avoid TCP sequence reuse"
                )
            path = output / (name + ".pcap")
            duration = 8 * count["frame_bytes"] / job["rate_bps"]
            traffic.setdefault(profile, []).append(
                {
                    **{
                        field: job[field]
                        for field in (
                            "id",
                            "source",
                            "target",
                            "start_seconds",
                            "rate_bps",
                        )
                    },
                    "kind": "pcap",
                    "path": path.name,
                    "sha256": _sha256(path),
                    "provenance": {
                        "capture": capture.name,
                        "capture_sha256": capture_sha256,
                        "flow": recipe["flow"],
                        "first_frame": job["first_frame"],
                        "last_frame": job["last_frame"],
                        "recipe": recipe.get("provenance", recipe_path.name),
                        **count,
                        "estimated_replay_seconds": round(duration, 3),
                    },
                }
            )
        scenario_file.write_text(json.dumps(scenario.data, indent=2) + "\n")
    except BaseException:
        scenario_file.unlink(missing_ok=True)
        for _, job, _ in jobs:
            (output / (job["id"] + ".pcap")).unlink(missing_ok=True)
        output.rmdir()
        raise
    return scenario_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="Flow and replay windows JSON (default: scenarios/corpora-example.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for scenario.json and replay files",
    )
    args = parser.parse_args()
    try:
        scenario = prepare(args.capture, args.scenario, args.recipe, args.output)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.error(str(error))
    print(scenario)


if __name__ == "__main__":
    main()
