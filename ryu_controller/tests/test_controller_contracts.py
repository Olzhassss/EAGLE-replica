"""Topology, forwarding, and counter calculations without live switches."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ryu.ofproto import (
    ofproto_v1_3,
    ofproto_v1_3_parser,
    ofproto_v1_5,
    ofproto_v1_5_parser,
)
from ryu_controller.openflow_rules import (
    COOKIE,
    apply_forwarding_rules,
    request_master,
    request_slave,
)
from ryu_controller.routing import plan_forwarding_rules
from ryu_controller.telemetry import Telemetry, compute_link_utilization, counter_delta
from ryu_controller.topology_state import TopologyState, load_scenario

SOURCE = Path(__file__).resolve().parents[2] / "scenarios/v0.json"


class ControllerContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = load_scenario(SOURCE)
        self.topology = TopologyState(self.scenario)
        ports = {
            name: list(self.scenario.ports[name]) for name in self.scenario.switches
        }
        self.frames = {slot: {"ports": ports} for slot in self.scenario.controllers}
        self.topology.update_presence(self.frames)

    def test_active_routing_removes_support_while_configured_observation_keeps_it(
        self,
    ) -> None:
        self.topology.set_support_action("c0", 0, [1, "c0"])
        self.topology.update_presence(self.frames)
        self.assertNotIn("s12", self.topology.active)
        self.assertIn("s12", self.topology.observation_graph("c0", "configured"))
        rules = plan_forwarding_rules(self.topology.active, self.scenario)
        self.assertEqual(rules["s12"], {})
        self.assertTrue(rules["s6"])

    def test_controller_failure_changes_owner_without_reordering_slots(self) -> None:
        del self.frames["c0"]
        self.topology.update_presence(self.frames)
        self.assertEqual(self.topology.owners["c0"], "c1")
        self.assertIn("s12", self.topology.active)
        self.assertEqual(tuple(self.topology.resources), ("c0", "c1", "c2"))

    def test_counter_rates_and_reset(self) -> None:
        self.assertEqual(counter_delta(5, 10), 5)
        self.assertEqual(counter_delta(11, 10), 1)
        self.assertEqual(compute_link_utilization(125, 2, 1000), 0.5)
        self.assertEqual(compute_link_utilization(1, 2, 1000), 0.0)

    def test_serialize_switch_commands_for_both_versions(self) -> None:
        for ofp, parser in (
            (ofproto_v1_3, ofproto_v1_3_parser),
            (ofproto_v1_5, ofproto_v1_5_parser),
        ):
            with self.subTest(version=ofp.OFP_VERSION):
                datapath = Mock(ofproto=ofp, ofproto_parser=parser)
                request_master(datapath, 1)
                request_slave(datapath, 2)
                macs = ("00:00:00:00:00:01", "00:00:00:00:00:02")
                apply_forwarding_rules(datapath, None, {macs: 1})
                for call in datapath.send_msg.call_args_list:
                    message = call[0][0]
                    message.serialize()
                    self.assertEqual(message.buf[0], ofp.OFP_VERSION)

    def test_openflow_15_reads_oxs_byte_count(self) -> None:
        parser = ofproto_v1_5_parser
        datapath = Mock(ofproto=ofproto_v1_5, ofproto_parser=parser)
        xids = iter((1, 2))
        datapath.set_xid.side_effect = lambda request: request.set_xid(next(xids))
        source, target = list(self.scenario.hosts)[:2]
        switch = next(iter(self.scenario.graph[source]))
        telemetry = Telemetry(self.scenario)
        telemetry.request_round({switch: datapath}, generation=1)
        port_request, flow_request = (
            call[0][0] for call in datapath.send_msg.call_args_list
        )
        self.assertIsInstance(flow_request, parser.OFPFlowDescStatsRequest)
        flow_request.serialize()
        self.assertEqual(flow_request.cookie, COOKIE)
        match = parser.OFPMatch(
            eth_src=self.scenario.hosts[source]["mac"],
            eth_dst=self.scenario.hosts[target]["mac"],
        )
        stats = parser.OFPStats(byte_count=42)
        entry = parser.OFPFlowDesc(cookie=COOKIE, match=match, stats=stats)
        flow_reply = parser.OFPFlowDescStatsReply(datapath, body=[entry], flags=0)
        flow_reply.set_xid(flow_request.xid)
        telemetry.receive(flow_reply)

        port = next(iter(self.scenario.ports[switch]))
        counters = SimpleNamespace(port_no=port, rx_bytes=50, tx_bytes=60)
        port_reply = parser.OFPPortStatsReply(datapath, body=[counters], flags=0)
        port_reply.set_xid(port_request.xid)
        telemetry.receive(port_reply)
        self.assertEqual(telemetry.previous["flows"][switch], {(source, target): 42})
        self.assertEqual(telemetry.previous["ports"][switch], {port: [50, 60]})
