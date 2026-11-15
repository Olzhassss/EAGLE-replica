"""Topology, forwarding, and counter calculations without live switches."""

from pathlib import Path
import unittest

from ryu_controller.routing import plan_forwarding_rules
from ryu_controller.telemetry import compute_link_utilization, counter_delta
from ryu_controller.topology_state import TopologyState, load_scenario


SOURCE = Path(__file__).resolve().parents[2] / "scenarios/v0.json"


class ControllerContracts(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(SOURCE)
        self.topology = TopologyState(self.scenario)
        ports = {name: list(self.scenario.ports[name]) for name in self.scenario.switches}
        self.frames = {slot: {"ports": ports} for slot in self.scenario.controllers}
        self.topology.update_presence(self.frames)

    def test_active_routing_removes_support_while_configured_observation_keeps_it(self):
        self.topology.set_support_action("c0", 0, [1, "c0"])
        self.topology.update_presence(self.frames)
        self.assertNotIn("s12", self.topology.active)
        self.assertIn("s12", self.topology.observation_graph("c0", "configured"))
        rules = plan_forwarding_rules(self.topology.active, self.scenario)
        self.assertEqual(rules["s12"], {})
        self.assertTrue(rules["s6"])

    def test_controller_failure_changes_owner_without_reordering_slots(self):
        del self.frames["c0"]
        self.topology.update_presence(self.frames)
        self.assertEqual(self.topology.owners["c0"], "c1")
        self.assertIn("s12", self.topology.active)
        self.assertEqual(tuple(self.topology.resources), ("c0", "c1", "c2"))

    def test_counter_rates_and_reset(self):
        self.assertEqual(counter_delta(5, 10), 5)
        self.assertEqual(counter_delta(11, 10), 1)
        self.assertEqual(compute_link_utilization(125, 2, 1000), 0.5)
        self.assertEqual(compute_link_utilization(1, 2, 1000), 0.0)
