"""Network validation without starting Mininet."""

import json
import unittest
from pathlib import Path

from mininet.network import validate_network


SOURCE = (
    Path(__file__).resolve().parents[2] / "EAGLE/src/network_preparation/source_v0.json"
)


class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = json.loads(SOURCE.read_text())

    def test_source_manifest(self) -> None:
        validate_network(self.network)

    def test_system_command_parameters_require_positive_integers(self) -> None:
        for row, key in (
            (self.network["controllers"][0], "openflow_port"),
            (self.network["switches"][0], "dpid"),
            (self.network["links"][0], "source_port"),
            (self.network["links"][0], "capacity_bps"),
        ):
            original = row[key]
            for invalid in (True, 1.5, float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    row[key] = invalid
                    with self.assertRaises(ValueError):
                        validate_network(self.network)
            row[key] = original

    def test_shell_metacharacters_are_rejected_in_node_names(self) -> None:
        self.network["switches"][0]["id"] = "s1;"
        with self.assertRaisesRegex(ValueError, "node name"):
            validate_network(self.network)
