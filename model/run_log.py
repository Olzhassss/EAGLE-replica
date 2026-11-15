"""Persist experiment settings and event records before generating figures."""

import json
from pathlib import Path
import platform
import shutil
from importlib.metadata import version


class RunLog:
    def __init__(self, directory, scenario, settings, **metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        manifest = dict(
            metadata, schema_version=1, scenario_path=str(scenario.path), scenario=scenario.data,
            scenario_sha256=scenario.fingerprint, settings=settings, python=platform.python_version(),
            libraries={name: version(name) for name in ("torch", "numpy", "networkx", "scipy", "gymnasium")},
        )
        (self.directory / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        self.events = self.directory / "events.jsonl"
        self.capture_cache = {}

    def write(self, event, **fields):
        with self.events.open("a") as stream:
            stream.write(json.dumps(dict(fields, event=event), allow_nan=False) + "\n")

    def archive_captures(self, episode, traffic):
        for job in traffic.get("flows", []):
            for name in ("log", "sent_capture", "received_capture", "sender_record", "receiver_record"):
                if name in job and Path(job[name]).exists():
                    destination = self.directory / "traffic" / str(episode)
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(job[name], destination)


def read_events(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_calibration(path):
    for row in read_events(path):
        if row["event"] in ("reset", "transition"):
            yield row["raw_embedding"]
