"""Load network definitions shared by training and evaluation."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from EAGLE.src.paths import ROOT
from network_schema import validate_support_layout


@dataclass(frozen=True)
class Network:
    """A network definition, its source path, and the digest used by services."""

    path: Path
    data: dict
    fingerprint: str

    @property
    def controller_ids(self) -> tuple[str, ...]:
        return tuple(row["id"] for row in self.data["controllers"])

    @property
    def support_counts(self) -> tuple[int, ...]:
        return tuple(len(row["support_switches"]) for row in self.data["controllers"])

    @property
    def action_sizes(self) -> tuple[int, ...]:
        return tuple(1 << count for count in self.support_counts)

    @property
    def all_on_actions(self) -> tuple[int, ...]:
        return tuple(size - 1 for size in self.action_sizes)


def load_network(path: Path | str) -> Network:
    """Read a JSON path or a filename from the repository's networks directory."""
    if isinstance(path, str) and Path(path).name == path:
        path = ROOT / "networks" / path
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    validate_support_layout(data)
    return Network(path, data, hashlib.sha256(raw).hexdigest())
