"""Persist experiment settings and event records before generating figures."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .scenario import Scenario


class RunLog:
    def __init__(
        self,
        directory: Path | str,
        scenario: Scenario,
        settings: dict[str, Any],
        *,
        resume: bool = False,
        overwrite: bool = False,
        **metadata: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "run.json"
        if manifest_path.exists() and not (resume or overwrite):
            raise FileExistsError(f"Run already exists in {self.directory}")
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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )
        self.events = self.directory / "events.jsonl"
        if overwrite:
            self.events.write_text("")
        self.capture_cache = {}

    def write(self, event: str, **fields: Any) -> None:
        with self.events.open("a") as stream:
            stream.write(json.dumps(dict(fields, event=event), allow_nan=False) + "\n")
