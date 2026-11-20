"""Replay prepared jobs in host namespaces and retain delivery measurements."""

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

FLOW_SCRIPT = Path(__file__).with_name("traffic_flow.py").resolve()


class TrialTraffic:
    def __init__(self, network, session_dir: Path, owner_uid: int) -> None:
        self.network = network
        self.session_dir = session_dir
        self.owner_uid = owner_uid
        self.processes = []
        self.records = []
        self.directory = None
        self.hash_cache = {}
        self.measure_flows = True

    def _spawn(self, host, command: list, log: Path, role: str) -> None:
        with log.open("ab") as output:
            process = host.popen(
                list(map(str, command)), stdout=output, stderr=subprocess.STDOUT
            )
        self.processes.append((process, log, role))

    def _prepare(self, jobs: list[dict], duration: float) -> list[dict]:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Traffic duration must be positive")
        names = {host.name for host in self.network.hosts}
        identifiers = set()
        prepared = []
        for entry in jobs:
            job = dict(entry)
            name = job["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
                raise ValueError("Invalid or duplicate traffic job ID")
            identifiers.add(name)
            if {job["source"], job["target"]} - names or job["source"] == job["target"]:
                raise ValueError("Traffic needs distinct configured hosts")
            if not 0 <= job.get("start_seconds", 0) < duration:
                raise ValueError("Traffic start falls outside its duration")
            if job["kind"] == "tcp":
                job.setdefault("timeout_seconds", 60)
                if type(job["size_bytes"]) is not int:
                    raise ValueError("TCP size must be an integer byte count")
                if any(
                    not math.isfinite(job[key]) or job[key] <= 0
                    for key in ("size_bytes", "rate_bps", "timeout_seconds")
                ):
                    raise ValueError("TCP size, rate, and timeout must be positive")
            elif job["kind"] == "pcap":
                programs = (
                    ("tcpreplay", "tcpdump") if self.measure_flows else ("tcpreplay",)
                )
                for program in programs:
                    if shutil.which(program) is None:
                        raise RuntimeError("Install " + program + " for PCAP replay")
                path = Path(job["path"]).resolve()
                stat = path.stat()
                key = (path, stat.st_size, stat.st_mtime_ns)
                if key not in self.hash_cache:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    self.hash_cache[key] = digest.hexdigest()
                if self.hash_cache[key] != job["sha256"]:
                    raise ValueError("PCAP checksum differs from its manifest")
                job["resolved_path"] = str(path)
            else:
                raise ValueError("Unknown traffic kind")
            prepared.append(job)
        return prepared

    def start(
        self, jobs: list[dict], duration: float, *, measure_flows: bool = True
    ) -> dict:
        self.finish()
        if self.directory:
            shutil.rmtree(self.directory)
            self.directory = None
        self.records = []
        self.measure_flows = measure_flows
        jobs = self._prepare(jobs, duration)
        self.directory = Path(tempfile.mkdtemp(prefix="trial-", dir=self.session_dir))
        self.directory.chmod(0o711)
        senders = defaultdict(list)
        receivers = {}
        try:
            for target in sorted(
                {job["target"] for job in jobs if job["kind"] == "tcp"}
            ):
                directory = self.directory / target
                directory.mkdir()
                receivers[target] = directory
                self._spawn(
                    self.network[target],
                    [sys.executable, FLOW_SCRIPT, "serve", directory],
                    self.directory / (target + ".log"),
                    "receiver",
                )
            for job in jobs:
                record = dict(job, measure_flows=self.measure_flows)
                prefix = self.directory / job["id"]
                if job["kind"] == "tcp" and self.measure_flows:
                    record.update(
                        sender_record=str(prefix.with_suffix(".sent.json")),
                        receiver_record=str(
                            receivers[job["target"]] / (job["id"] + ".received.json")
                        ),
                    )
                elif job["kind"] == "pcap" and self.measure_flows:
                    outgoing = prefix.with_suffix(".out.pcap")
                    incoming = prefix.with_suffix(".in.pcap")
                    source, target = (
                        self.network[job["source"]],
                        self.network[job["target"]],
                    )
                    for host, capture in ((source, outgoing), (target, incoming)):
                        self._spawn(
                            host,
                            [
                                "tcpdump",
                                "-Z",
                                "root",
                                "-U",
                                "-s",
                                "0",
                                "-i",
                                host.defaultIntf().name,
                                "-w",
                                capture,
                                "ether",
                                "src",
                                source.MAC(),
                                "and",
                                "ether",
                                "dst",
                                target.MAC(),
                                "and",
                                "tcp",
                            ],
                            prefix.with_suffix(".log"),
                            "capture",
                        )
                    record.update(
                        sent_capture=str(outgoing), received_capture=str(incoming)
                    )
                self.records.append(record)
            deadline = time.monotonic() + 10
            while any(
                not (directory / "ready").exists() for directory in receivers.values()
            ):
                if time.monotonic() >= deadline or any(
                    process.poll() is not None for process, _, _ in self.processes
                ):
                    raise RuntimeError("Traffic receiver failed to start")
                time.sleep(0.05)
            time.sleep(0.2)
            if any(process.poll() is not None for process, _, _ in self.processes):
                raise RuntimeError("Traffic receiver or capture failed to start")
            started = time.time() + 1
            for record in self.records:
                record["scheduled_at"] = started + record.get("start_seconds", 0)
                source = self.network[record["source"]]
                if record["kind"] == "tcp":
                    senders[record["source"]].append(
                        dict(record, address=self.network[record["target"]].IP()),
                    )
                else:
                    self._spawn(
                        source,
                        [
                            sys.executable,
                            FLOW_SCRIPT,
                            "replay",
                            record["scheduled_at"],
                            "tcpreplay",
                            "--intf1=" + source.defaultIntf().name,
                            "--multiplier=1",
                            record["resolved_path"],
                        ],
                        self.directory / (record["id"] + ".log"),
                        "sender",
                    )
            for source, rows in senders.items():
                path = self.directory / (source + ".jobs.json")
                path.write_text(json.dumps(rows))
                self._spawn(
                    self.network[source],
                    [sys.executable, FLOW_SCRIPT, "batch", path, self.directory],
                    self.directory / (source + ".send.log"),
                    "sender",
                )
        except BaseException:
            self.finish()
            raise
        return {"started_at": started, "duration_seconds": duration}

    def finish(self, drain_seconds: float = 0) -> dict:
        if not math.isfinite(drain_seconds) or drain_seconds < 0:
            raise ValueError("Drain seconds must be nonnegative")
        if self.processes:
            time.sleep(drain_seconds)
        outcomes = []
        for process, log, role in sorted(
            self.processes, key=lambda row: row[2] != "sender"
        ):
            if process.poll() is None:
                process.send_signal(
                    signal.SIGINT if role == "capture" else signal.SIGTERM
                )
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            outcomes.append(
                {
                    "pid": process.pid,
                    "role": role,
                    "exit_code": process.returncode,
                    "log": str(log),
                }
            )
        self.processes = []
        if self.directory:
            for path in self.directory.rglob("*"):
                if path.is_file():
                    os.chown(path, self.owner_uid, -1)
                    path.chmod(0o600)
                else:
                    path.chmod(0o711)
        return {
            "flows": self.records if self.measure_flows else [],
            "processes": outcomes,
        }
