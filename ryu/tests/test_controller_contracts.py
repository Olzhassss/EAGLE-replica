"""Topology, forwarding, and counter calculations without live switches."""

import threading
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ryu.ofproto import (
    ofproto_v1_3,
    ofproto_v1_3_parser,
    ofproto_v1_5,
    ofproto_v1_5_parser,
)
from webob import Request

from ryu.api import EagleApi
from ryu.eagle_controller import EagleController
from ryu.network import load_network
from ryu.openflow_rules import (
    COOKIE,
    apply_forwarding_rules,
    request_master,
    request_slave,
)
from ryu.routing import plan_forwarding_rules
from ryu.telemetry import Telemetry, compute_link_utilization, counter_delta
from ryu.topology_state import TopologyState


SOURCE = (
    Path(__file__).resolve().parents[2] / "EAGLE/src/network_preparation/source_v0.json"
)


class ControllerContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.network = load_network(SOURCE)
        self.topology = TopologyState(self.network)
        ports = {name: list(self.network.ports[name]) for name in self.network.switches}
        self.frames = {slot: {"ports": ports} for slot in self.network.controllers}
        self.topology.update_presence(self.frames)

    def test_active_observation_and_routing_remove_disabled_support(self) -> None:
        self.topology.set_support_action("c0", 0, [1, "c0"])
        self.topology.update_presence(self.frames)
        self.assertNotIn("s12", self.topology.active)
        self.assertNotIn("s12", self.topology.observation_graph("c0"))
        rules = plan_forwarding_rules(self.topology.active, self.network)
        self.assertEqual(rules["s12"], {})
        self.assertTrue(rules["s6"])

    def test_controller_failure_changes_owner_without_reordering_slots(self) -> None:
        del self.frames["c0"]
        self.topology.update_presence(self.frames)
        self.assertEqual(self.topology.owners["c0"], "c1")
        self.assertIn("s12", self.topology.active)
        self.assertEqual(tuple(self.topology.resources), ("c0", "c1", "c2"))

    def test_support_bits_change_topology_routes_and_power_counts_independently(
        self,
    ) -> None:
        self.network.controllers["c0"]["support_switches"] = ["s12", "s10"]
        self.network.controllers["c0"]["initial_action"] = 3
        topology = TopologyState(self.network)
        topology.update_presence(self.frames)
        all_on = topology.resource_usage("c0")[0]
        for mask in range(4):
            topology.set_support_action("c0", mask, [mask + 1, "c0"])
            topology.update_presence(self.frames)
            rules = plan_forwarding_rules(topology.active, self.network)
            for index, node in enumerate(["s12", "s10"]):
                enabled = bool(mask & (1 << index))
                self.assertEqual(node in topology.active, enabled)
                self.assertEqual(node in topology.observation_graph("c0"), enabled)
                if not enabled:
                    self.assertEqual(rules[node], {})
            self.assertEqual(
                topology.resource_usage("c0")[0], all_on - 2 + bin(mask).count("1")
            )

    def test_multiple_support_actions_survive_peer_merge_and_failover(self) -> None:
        self.network.controllers["c0"]["support_switches"] = ["s12", "s10"]
        topology = TopologyState(self.network)
        topology.merge_resources({"c0": {"action": 2, "revision": [1, "c0"]}})
        topology.merge_resources({"c0": {"action": 4, "revision": [2, "c0"]}})
        self.assertEqual(topology.resources["c0"]["action"], 2)
        frames = {slot: frame for slot, frame in self.frames.items() if slot != "c0"}
        topology.update_presence(frames)
        self.assertEqual(topology.owners["c0"], "c1")
        self.assertNotIn("s12", topology.active)
        self.assertIn("s10", topology.active)
        with self.assertRaises(ValueError):
            topology.set_support_action("c0", -1, [3, "c0"])

    def test_http_accepts_mask_actions_and_rejects_out_of_range_values(self) -> None:
        self.network.controllers["c0"]["support_switches"] = ["s12", "s10"]
        self.network.controllers["c0"]["initial_action"] = 3
        topology = TopologyState(self.network)
        topology.update_presence(self.frames)
        api = EagleApi.__new__(EagleApi)
        api.app = SimpleNamespace(
            network=self.network,
            controller_id="c0",
            topology=topology,
            lock=threading.Lock(),
            telemetry=Mock(),
            sequence=0,
        )
        for payload in ([2], [4], [-1], [True], [1.5], [1, 0], {}):
            request = Request.blank(
                "/model/actions?controller_id=c0",
                method="PUT",
                content_type="application/json",
                body=json.dumps(payload).encode(),
            )
            response = api.put_actions(request)
            self.assertEqual(response.status_int, 200 if payload == [2] else 400)
        self.assertEqual(topology.resources["c0"]["action"], 2)
        self.assertEqual(api.app.sequence, 1)

    def test_counter_rates_and_reset(self) -> None:
        self.assertEqual(counter_delta(5, 10), 5)
        self.assertEqual(counter_delta(11, 10), 1)
        self.assertEqual(compute_link_utilization(125, 2, 1000), 0.5)
        self.assertEqual(compute_link_utilization(1, 2, 1000), 0.0)

    def test_empty_controller_publishes_a_coherent_poll_window(self) -> None:
        telemetry = Telemetry(self.network)
        with patch("ryu.telemetry.time.time", side_effect=[10.0, 11.0]):
            telemetry.request_round({}, generation=1)
            self.assertIsNone(telemetry.latest)
            telemetry.request_round({}, generation=1)
        self.assertIsNone(telemetry.pending)
        self.assertEqual(telemetry.latest["sample_duration_seconds"], 1)
        self.assertEqual(telemetry.latest["ports"], {})
        self.assertEqual(telemetry.latest["flows"], {})

    def test_http_action_acknowledges_the_network_identity(self) -> None:
        api = EagleApi.__new__(EagleApi)
        api.app = SimpleNamespace(
            network=self.network,
            controller_id="c0",
            topology=self.topology,
            lock=threading.Lock(),
            telemetry=Mock(),
            sequence=0,
        )
        request = Request.blank(
            "/model/actions?controller_id=c0&network=" + self.network.fingerprint,
            method="PUT",
            content_type="application/json",
            body=b"[0]",
        )
        response = api.put_actions(request)
        self.assertEqual(response.status_int, 200)
        self.assertEqual(response.json["network"], self.network.fingerprint)
        self.assertEqual(response.json["action"], 0)
        self.assertTrue(response.json["changed"])
        api.app.telemetry.reset.assert_called_once_with()

    def test_http_action_rejects_a_different_network(self) -> None:
        api = EagleApi.__new__(EagleApi)
        api.app = SimpleNamespace(network=self.network, controller_id="c0")
        request = Request.blank(
            "/model/actions?controller_id=c0&network=another-network",
            method="PUT",
            content_type="application/json",
            body=b"[0]",
        )
        self.assertEqual(api.put_actions(request).status_int, 409)

    def test_snapshot_counts_delivered_bytes_and_uses_poll_duration(self) -> None:
        edge = self.topology.active["s6"]["s9"]
        telemetry = Telemetry(self.network)
        telemetry.latest = {
            "sample_id": 1,
            "sampled_at": 3,
            "window_start": 1,
            "sample_duration_seconds": 2,
            "ports": {"s6": {edge["ports"]["s6"]: (0, edge["capacity_bps"] // 8)}},
            "flows": {"s6": {("h4", "h1"): 1024}, "s1": {("h4", "h1"): 512}},
            "port_counters": {},
            "flow_counters": {},
        }
        controller = SimpleNamespace(
            topology=self.topology,
            telemetry=telemetry,
            controller_id="c0",
            network=self.network,
            incarnation="test",
            rule_timing={},
        )
        snapshot = EagleController.stats(controller, "c0")
        self.assertEqual(snapshot["delivered_bytes"], {"h4 h1": 512})
        weights = {
            frozenset((row["source"], row["target"])): row["weight"]
            for row in snapshot["topology"]["links"]
        }
        self.assertEqual(weights[frozenset(("s6", "s9"))], 0.5)
        self.assertEqual(weights[frozenset(("s1", "s6"))], 0)
        self.assertFalse({"demands", "throughput", "stats"} & snapshot.keys())

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
        source, target = list(self.network.hosts)[:2]
        switch = next(iter(self.network.graph[source]))
        telemetry = Telemetry(self.network)
        telemetry.request_round({switch: datapath}, generation=1)
        port_request, flow_request = (
            call[0][0] for call in datapath.send_msg.call_args_list
        )
        self.assertIsInstance(flow_request, parser.OFPFlowDescStatsRequest)
        flow_request.serialize()
        self.assertEqual(flow_request.cookie, COOKIE)
        match = parser.OFPMatch(
            eth_src=self.network.hosts[source]["mac"],
            eth_dst=self.network.hosts[target]["mac"],
        )
        stats = parser.OFPStats(byte_count=42)
        entry = parser.OFPFlowDesc(cookie=COOKIE, match=match, stats=stats)
        flow_reply = parser.OFPFlowDescStatsReply(datapath, body=[entry], flags=0)
        flow_reply.set_xid(flow_request.xid)
        telemetry.receive(flow_reply)

        port = next(iter(self.network.ports[switch]))
        counters = SimpleNamespace(port_no=port, rx_bytes=50, tx_bytes=60)
        port_reply = parser.OFPPortStatsReply(datapath, body=[counters], flags=0)
        port_reply.set_xid(port_request.xid)
        telemetry.receive(port_reply)
        self.assertEqual(telemetry.previous["flows"][switch], {(source, target): 42})
        self.assertEqual(telemetry.previous["ports"][switch], {port: [50, 60]})
