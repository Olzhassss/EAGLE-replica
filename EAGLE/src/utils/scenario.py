"""Load shared topology and traffic scenarios for the RL project."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "networks/v0.json"


@dataclass(frozen=True)
class Scenario:
    path: Path
    data: dict
    fingerprint: str

    @property
    def controller_ids(self) -> tuple[str, ...]:
        return tuple(row["id"] for row in self.data["controllers"])

    @property
    def initial_actions(self) -> tuple[int, ...]:
        return tuple(row["initial_action"] for row in self.data["controllers"])


def load_scenario(path: Path | str) -> Scenario:
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError(f"Expected schema_version 1 in {path}")
    scenario = Scenario(path, data, hashlib.sha256(raw).hexdigest())
    if not scenario.controller_ids or len(set(scenario.controller_ids)) != len(
        scenario.controller_ids,
    ):
        raise ValueError(
            "The scenario needs distinct controller IDs in model slot order",
        )
    return scenario
