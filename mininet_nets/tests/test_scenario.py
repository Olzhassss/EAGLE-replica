"""Scenario validation without starting Mininet."""

import json
import unittest
from pathlib import Path

from mininet_nets.launch_topology import validate_scenario

SOURCE = Path(__file__).resolve().parents[2] / "scenarios/v0.json"


class ScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = json.loads(SOURCE.read_text())

    def test_source_manifest(self) -> None:
        validate_scenario(self.scenario)

    def test_shell_metacharacters_are_rejected_in_node_names(self) -> None:
        self.scenario["switches"][0]["id"] = "s1;"
        with self.assertRaisesRegex(ValueError, "node name"):
            validate_scenario(self.scenario)
