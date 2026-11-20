"""OpenFlow counter samples, coherent poll windows, and rate conversion."""

import time
from typing import Any, Dict, Union

from ryu.controller.controller import Datapath
from ryu.ofproto import ofproto_v1_3_parser, ofproto_v1_5, ofproto_v1_5_parser

from ryu_controller.openflow_rules import COOKIE, COOKIE_MASK, TABLE_ID

from .topology_state import Scenario

StatsReply = Union[
    ofproto_v1_3_parser.OFPFlowStatsReply,
    ofproto_v1_3_parser.OFPPortStatsReply,
    ofproto_v1_5_parser.OFPFlowDescStatsReply,
    ofproto_v1_5_parser.OFPPortStatsReply,
]


class Telemetry:
    """Correlate multipart replies within a single round and topology revision."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.pending = None
        self.previous = None
        self.latest = None
        self.sample_id = 0
        self.host_for_mac = {
            row["mac"].lower(): name for name, row in scenario.hosts.items()
        }

    def reset(self) -> None:
        self.pending = self.previous = self.latest = None

    def request_round(self, datapaths: Dict[str, Datapath], generation: int) -> None:
        started = time.time()
        self.pending = {
            "started_at": started,
            "generation": generation,
            "requests": {},
            "ports": {},
            "flows": {},
        }
        if not datapaths:
            return
        commands = []
        for switch, datapath in sorted(datapaths.items()):
            ofp = datapath.ofproto
            parser = datapath.ofproto_parser
            port_request = parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
            commands.append((switch, "ports", port_request))
            if any(node in self.scenario.hosts for node in self.scenario.graph[switch]):
                # OVS exposes per-flow counters in flow descriptions in 1.5.
                request_type = (
                    parser.OFPFlowDescStatsRequest
                    if ofp.OFP_VERSION == ofproto_v1_5.OFP_VERSION
                    else parser.OFPFlowStatsRequest
                )
                flow_request = request_type(
                    datapath,
                    0,
                    TABLE_ID,
                    ofp.OFPP_ANY,
                    ofp.OFPG_ANY,
                    COOKIE,
                    COOKIE_MASK,
                    parser.OFPMatch(),
                )
                commands.append((switch, "flows", flow_request))
        for switch, kind, request in commands:
            request.datapath.set_xid(request)
            key = (request.datapath, request.xid)
            self.pending["requests"][key] = (switch, kind, [])
        for _, _, request in commands:
            request.datapath.send_msg(request)

    def receive(self, message: StatsReply) -> None:
        poll = self.pending
        key = (message.datapath, message.xid)
        if poll is None or key not in poll["requests"]:
            return
        switch, kind, parts = poll["requests"][key]
        parts.extend(message.body)
        if message.flags & message.datapath.ofproto.OFPMPF_REPLY_MORE:
            return
        if kind == "ports":
            poll[kind][switch] = {
                stat.port_no: [stat.rx_bytes, stat.tx_bytes]
                for stat in parts
                if stat.port_no in self.scenario.ports[switch]
                and max(stat.rx_bytes, stat.tx_bytes) < (1 << 64) - 1
            }
        else:
            counters = {}
            for stat in parts:
                if stat.cookie != COOKIE:
                    continue
                if message.datapath.ofproto.OFP_VERSION == ofproto_v1_5.OFP_VERSION:
                    byte_count = stat.stats.get("byte_count")
                    if byte_count is None:
                        continue
                else:
                    byte_count = stat.byte_count
                source = self.host_for_mac.get(stat.match.get("eth_src"))
                target = self.host_for_mac.get(stat.match.get("eth_dst"))
                if not source or not target:
                    continue
                if (
                    target in self.scenario.graph[switch]
                    or source in self.scenario.graph[switch]
                ):
                    counters[(source, target)] = byte_count
            poll[kind][switch] = counters
        del poll["requests"][key]
        if not poll["requests"]:
            self.pending = None
            self._publish(poll)

    def _publish(self, current: Dict[str, Any]) -> None:
        previous = self.previous
        self.previous = current
        if previous is None or previous["generation"] != current["generation"]:
            return
        elapsed = current["started_at"] - previous["started_at"]
        if elapsed <= 0:
            return
        ports = {}
        for switch, counters in current["ports"].items():
            old = previous["ports"].get(switch, {})
            ports[switch] = {
                port: [
                    counter_delta(new, before) for new, before in zip(values, old[port])
                ]
                for port, values in counters.items()
                if port in old
            }
        flows = {}
        for switch, counters in current["flows"].items():
            old = previous["flows"].get(switch, {})
            flows[switch] = {
                pair: counter_delta(value, old[pair])
                for pair, value in counters.items()
                if pair in old
            }
        self.sample_id += 1
        self.latest = {
            "sample_id": self.sample_id,
            "window_start": previous["started_at"],
            "sampled_at": current["started_at"],
            "sample_duration_seconds": elapsed,
            "ports": ports,
            "flows": flows,
            "port_counters": current["ports"],
            "flow_counters": {
                node: {" ".join(pair): value for pair, value in rows.items()}
                for node, rows in current["flows"].items()
            },
        }


def counter_delta(value: int, previous: int) -> int:
    """Treat a lower counter as a reset at or before the current sample."""
    return value - previous if value >= previous else value


def compute_link_utilization(
    byte_delta: int,
    duration_seconds: float,
    capacity_bps: int,
) -> float:
    """Convert one direction's byte delta to the bounded utilization weight."""
    ratio = byte_delta * 8 / duration_seconds / capacity_bps
    return 0.0 if ratio < 0.01 else min(ratio, 1.0)
