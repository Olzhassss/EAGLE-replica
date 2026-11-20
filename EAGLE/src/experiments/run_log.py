"""Persist experiment settings and event records before generating figures."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from EAGLE.src.network import Network


class RunLog:
    """Persist an experiment manifest, append-only events, and a capture analysis cache."""

    def __init__(
        self,
        directory: Path | str,
        network: Network,
        settings: dict[str, Any],
        *,
        resume: bool = False,
        overwrite: bool = False,
        **metadata: Any,
    ) -> None:
        """Record network identity and settings before the first event is written."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "run.json"
        if manifest_path.exists() and not (resume or overwrite):
            raise FileExistsError(f"Run already exists in {self.directory}")
        manifest = dict(
            metadata,
            schema_version=1,
            network_path=str(network.path),
            network=network.data,
            network_sha256=network.fingerprint,
            settings=settings,
            python=platform.python_version(),
            libraries={
                name: version(name)
                for name in ("torch", "numpy", "networkx", "scipy", "gymnasium")
            },
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )
        self.events = self.directory / "events.jsonl"
        if overwrite:
            self.events.write_text("")
        self.capture_cache = {}

    def write(self, event: str, **fields: Any) -> None:
        """Append a finite JSON event without replacing earlier trial records."""
        with self.events.open("a") as stream:
            stream.write(json.dumps(dict(fields, event=event), allow_nan=False) + "\n")
