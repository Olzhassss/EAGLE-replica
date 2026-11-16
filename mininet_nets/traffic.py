"""Start scenario traffic in host namespaces and retain per-flow evidence."""

import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mininet.net import Mininet
from mininet.node import Host

FLOW_SCRIPT = Path(__file__).with_name("traffic_flow.py").resolve()


class TrialTraffic:
    def __init__(
        self,
        network: Mininet,
        profiles: dict[str, list[dict[str, Any]]],
        scenario_dir: Path,
        session_dir: Path,
        owner_uid: int,
    ) -> None:
        self.network = network
        self.profiles = profiles
        self.scenario_dir = scenario_dir
        self.session_dir = session_dir
        self.owner_uid = owner_uid
        self.processes = []
        self.records = []
        self.hash_cache = {}

    def _spawn(
        self,
        host: Host,
        command: Sequence[str | int | float | Path],
        log: Path,
        capture: bool = False,
    ) -> subprocess.Popen:
        with log.open("ab") as output:
            process = host.popen(
                list(map(str, command)),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.processes.append((process, log, capture))
        return process

    def _prepare(self, profile: str) -> list[dict[str, Any]]:
        entries = self.profiles[profile]
        hosts = {host.name for host in self.network.hosts}
        identifiers = set()
        prepared = []
        for job in entries:
            job = dict(job)
            name = job["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
                raise ValueError("Invalid or duplicate traffic job ID: " + name)
            identifiers.add(name)
            if {job["source"], job["target"]} - hosts or job["source"] == job["target"]:
                raise ValueError("Traffic jobs need two distinct scenario hosts")
            if job.get("start_seconds", 0) < 0:
                raise ValueError("Traffic start must be nonnegative")
            if job["kind"] == "tcp":
                if (
                    min(
                        job["size_bytes"],
                        job["rate_bps"],
                        job.get("timeout_seconds", 60),
                    )
                    <= 0
                ):
                    raise ValueError("TCP size, rate and timeout must be positive")
            elif job["kind"] == "pcap":
                for program in ("tcpreplay", "tcpdump"):
                    if shutil.which(program) is None:
                        raise RuntimeError("Install " + program + " for PCAP trials")
                file = (self.scenario_dir / job["path"]).resolve()
                metadata = file.stat()
                key = (file, metadata.st_size, metadata.st_mtime_ns)
                if key not in self.hash_cache:
                    digest = hashlib.sha256()
                    with file.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    self.hash_cache[key] = digest.hexdigest()
                if self.hash_cache[key] != job["sha256"] or job["rate_bps"] <= 0:
                    raise ValueError("PCAP hash or replay rate is invalid for " + name)
                job["resolved_path"] = str(file)
            else:
                raise ValueError("Unknown traffic kind " + job["kind"])
            prepared.append(job)
        return prepared

    def start(self, profile: str) -> dict[str, Any]:
        # fmt: off
        self.finish()
        jobs = self._prepare(profile)
        directory = Path(tempfile.mkdtemp(prefix="trial-", dir=str(self.session_dir)))
        directory.chmod(0o711)
        started_at = time.time() + 1.0
        self.records = []
        try:
            senders = []
            for index, job in enumerate(jobs):
                source, target = (
                    self.network[job["source"]],
                    self.network[job["target"]],
                )
                prefix = directory / job["id"]
                log = prefix.with_suffix(".log")
                starts_at = started_at + job.get("start_seconds", 0)
                record = dict(job, scheduled_at=starts_at, log=str(log))
                if job["kind"] == "tcp":
                    port = 20000 + index
                    timeout = job.get("timeout_seconds", 60)
                    sent = prefix.with_suffix(".sent.json")
                    received = prefix.with_suffix(".received.json")
                    self._spawn(
                        target,
                        [
                            sys.executable,FLOW_SCRIPT,"receive",port,job["size_bytes"],
                            timeout + job.get("start_seconds", 0) + 5,received
                        ],
                        log,
                    )
                    command = [
                        sys.executable,FLOW_SCRIPT,"send",
                        target.IP(),port,job["size_bytes"],
                        job["rate_bps"],starts_at,timeout,sent
                    ]
                    senders.append((source, command, log))
                    record.update(
                        sender_record=str(sent), receiver_record=str(received)
                    )
                else:
                    outgoing = prefix.with_suffix(".out.pcap")
                    incoming = prefix.with_suffix(".in.pcap")
                    for host, capture in ((source, outgoing), (target, incoming)):
                        self._spawn(
                            host,
                            [
                                "tcpdump","-Z","root","-U","-s","0",
                                "-i",host.defaultIntf().name,"-w",capture,"ether",
                                "src",source.MAC(),"and","ether","dst",target.MAC(),"and","tcp"
                            ],
                            log,
                            capture=True,
                        )
                    command = [
                        sys.executable,FLOW_SCRIPT,"replay",starts_at,"tcpreplay",
                        "--intf1=" + source.defaultIntf().name,
                        "--mbps=" + str(job["rate_bps"] / 1_000_000),job["resolved_path"]
                    ]
                    senders.append((source, command, log))
                    record.update(
                        sent_capture=str(outgoing), received_capture=str(incoming)
                    )
                self.records.append(record)
            if jobs:
                time.sleep(0.2)
            for process, log, _ in self.processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Traffic receiver or capture exited during startup. See "
                        + str(log)
                    )
            for source, command, log in senders:
                self._spawn(source, command, log)
        except BaseException:
            self.finish()
            raise
        return {"started_at": started_at, "flows": self.records}
        # fmt: on

    def finish(self) -> dict[str, Any]:
        for process, _, capture in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT if capture else signal.SIGTERM)
        outcomes = []
        for process, log, capture in self.processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            outcomes.append(
                {
                    "pid": process.pid,
                    "log": str(log),
                    "capture": capture,
                    "exit_code": process.returncode,
                }
            )
        self.processes = []
        for job in self.records:
            for kind in (
                "log",
                "sent_capture",
                "received_capture",
                "sender_record",
                "receiver_record",
            ):
                if kind in job:
                    file = Path(job[kind])
                    if file.exists():
                        os.chown(file, self.owner_uid, -1)
                        file.chmod(0o600)
        return {"flows": self.records, "processes": outcomes}
