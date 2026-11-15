"""Scenario validation without starting Mininet."""

import json
from pathlib import Path
import unittest

from mininet_nets.launch_topology import validate_scenario


SOURCE = Path(__file__).resolve().parents[2] / "scenarios/v0.json"


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(SOURCE.read_text())

    def test_source_manifest(self):
        validate_scenario(self.scenario)

    def test_shell_metacharacters_are_rejected_in_node_names(self):
        self.scenario["switches"][0]["id"] = "s1;"
        with self.assertRaisesRegex(ValueError, "node name"):
            validate_scenario(self.scenario)
