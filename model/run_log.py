"""Persist experiment settings and event records before generating figures."""

import json
import platform
import shutil
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .settings import Scenario


class RunLog:
    def __init__(
        self,
        directory: Path | str,
        scenario: Scenario,
        settings: dict[str, Any],
        **metadata: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        manifest = dict(
            metadata,
            schema_version=1,
            scenario_path=str(scenario.path),
            scenario=scenario.data,
            scenario_sha256=scenario.fingerprint,
            settings=settings,
            python=platform.python_version(),
            libraries={
                name: version(name)
                for name in ("torch", "numpy", "networkx", "scipy", "gymnasium")
            },
        )
        (self.directory / "run.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )
        self.events = self.directory / "events.jsonl"
        self.capture_cache = {}

    def write(self, event: str, **fields: Any) -> None:
        with self.events.open("a") as stream:
            stream.write(json.dumps(dict(fields, event=event), allow_nan=False) + "\n")

    def archive_captures(self, episode: int, traffic: dict[str, Any]) -> None:
        for job in traffic.get("flows", []):
            for name in (
                "log",
                "sent_capture",
                "received_capture",
                "sender_record",
                "receiver_record",
            ):
                if name in job and Path(job[name]).exists():
                    destination = self.directory / "traffic" / str(episode)
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(job[name], destination)


def read_events(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_calibration(path: Path | str) -> Iterator[list[float]]:
    for row in read_events(path):
        if row["event"] in ("reset", "transition"):
            yield row["raw_embedding"]
