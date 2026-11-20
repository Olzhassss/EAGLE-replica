"""Own OpenFlow routing, peer state exchange, and coherent telemetry."""

import json
import os
import time
from typing import Any, Dict, Optional, Union
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from ryu.api import EagleApi
from ryu.app.wsgi import WSGIApplication
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.controller import Datapath
from ryu.controller.handler import DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.network import load_network
from ryu.ofproto import ofproto_v1_3, ofproto_v1_5
from ryu.openflow_rules import (
    apply_forwarding_rules,
    port_is_up,
    request_master,
    request_slave,
)
from ryu.routing import ForwardingRules, plan_forwarding_rules
from ryu.telemetry import Telemetry, compute_link_utilization
from ryu.topology_state import TopologyState

OPENFLOW_VERSION = "1.5"


class EagleController(app_manager.RyuApp):
    """Own OpenFlow datapaths, peer state, forwarding rules, and telemetry for one controller."""

    OFP_VERSIONS = [  # noqa: RUF012
        {"1.3": ofproto_v1_3.OFP_VERSION, "1.5": ofproto_v1_5.OFP_VERSION}[
            OPENFLOW_VERSION
        ],
    ]
    _CONTEXTS = {"wsgi": WSGIApplication}  # noqa: RUF012

    def __init__(self, *args, **kwargs) -> None:
        """Load the network and start independent telemetry and peer reconciliation loops."""
        super().__init__(*args, **kwargs)
        self.network = load_network(os.environ["EAGLE_NETWORK"])
        self.controller_id = os.environ["EAGLE_CONTROLLER_ID"]
        self.incarnation = time.time_ns()
        self.sequence = 0
        self.topology = TopologyState(self.network)
        self.telemetry = Telemetry(self.network)
        self.switch_for_dpid = {
            row["dpid"]: node for node, row in self.network.switches.items()
        }
        self.datapaths = {}
        self.port_states = {}
        self.master = set()
        self.requested_roles = {}
        self.installed = {}
        self.seen_peers = {self.controller_id}
        self.seen_switches = set()
        self.peers = {}
        self.topology_key = None
        self.generation = 0
        self.next_poll = 0.0
        self.pending_barriers = {}
        self.rule_timing = {}
        self.route_error = None
        self.lock = hub.Semaphore()
        self.http = build_opener(ProxyHandler({}))
        kwargs["wsgi"].register(EagleApi, {"app": self})

    def start(self) -> None:
        super().start()
        self.threads.append(hub.spawn(self._run))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def on_datapath_state(self, event: ofp_event.EventOFPStateChange) -> None:
        """Track connected datapaths and reconcile local forwarding ownership."""
        datapath = event.datapath
        switch = self.switch_for_dpid.get(datapath.id)
        if switch is None:
            return
        with self.lock:
            if event.state == MAIN_DISPATCHER:
                expected = self.network.ports[switch].keys()
                missing = expected - datapath.ports.keys()
                if missing:
                    self.logger.error(
                        "%s is missing configured OpenFlow ports %s",
                        switch,
                        sorted(missing),
                    )
                    return
                self.datapaths[switch] = datapath
                self.port_states[switch] = {
                    port
                    for port in expected
                    if port_is_up(datapath, datapath.ports[port])
                }
                self.seen_switches.add(switch)
                self._forget_programming(switch)
            elif self.datapaths.get(switch) is datapath:
                del self.datapaths[switch]
                del self.port_states[switch]
                self._forget_programming(switch)

    def _forget_programming(self, switch: str) -> None:
        self.requested_roles.pop(switch, None)
        self.master.discard(switch)
        self.installed.pop(switch, None)
        self.pending_barriers.pop(switch, None)
        self.topology_key = None
        self.telemetry.reset()

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def on_port_status(self, event: ofp_event.EventOFPPortStatus) -> None:
        """Refresh observed ports when a switch reports an interface change."""
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        port = message.desc.port_no
        with self.lock:
            if (
                self.datapaths.get(switch) is not message.datapath
                or port not in self.network.ports[switch]
            ):
                return
            if message.reason == message.datapath.ofproto.OFPPR_DELETE or not port_is_up(
                message.datapath, message.desc
            ):
                self.port_states[switch].discard(port)
            else:
                self.port_states[switch].add(port)

    @set_ev_cls(ofp_event.EventOFPRoleReply, MAIN_DISPATCHER)
    def on_role_reply(self, event: ofp_event.EventOFPRoleReply) -> None:
        """Record the switch-confirmed role before accepting master ownership."""
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        with self.lock:
            if self.datapaths.get(switch) is not message.datapath:
                return
            if message.role == message.datapath.ofproto.OFPCR_ROLE_MASTER:
                self.master.add(switch)
            else:
                self.master.discard(switch)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def on_barrier_reply(self, event: ofp_event.EventOFPBarrierReply) -> None:
        """Complete a tracked programming batch after the switch acknowledges its barrier."""
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        with self.lock:
            if self.pending_barriers.get(switch) == (message.datapath, message.xid):
                del self.pending_barriers[switch]
                if not self.pending_barriers:
                    self.rule_timing["barrier_at"] = time.time()

    @set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
    def on_openflow_error(self, event: ofp_event.EventOFPErrorMsg) -> None:
        """Retire failed command batches and invalidate the affected telemetry round."""
        message = event.msg
        switch: str = self.switch_for_dpid.get(message.datapath.id)
        self.logger.warning(
            "OpenFlow error on %s: type=%s code=%s",
            switch,
            message.type,
            message.code,
        )
        with self.lock:
            if self.datapaths.get(switch) is message.datapath and (
                switch in self.master or self.requested_roles.get(switch, (False, 0))[0]
            ):
                self._forget_programming(switch)
                self.route_error = "OpenFlow programming rejected on " + switch

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    @set_ev_cls(ofp_event.EventOFPFlowDescStatsReply, MAIN_DISPATCHER)
    def on_stats(
        self,
        event: Union[
            ofp_event.EventOFPPortStatsReply,
            ofp_event.EventOFPFlowStatsReply,
            ofp_event.EventOFPFlowDescStatsReply,
        ],
    ) -> None:
        """Pass multipart replies to the active poll round for this datapath."""
        with self.lock:
            self.telemetry.receive(event.msg)

    def presence(self) -> Dict[str, Any]:
        """Describe local ports, action revisions, and network identity for peer reconciliation."""
        return {
            "schema_version": 1,
            "network": self.network.fingerprint,
            "controller_id": self.controller_id,
            "incarnation": self.incarnation,
            "sequence": self.sequence,
            "owners": self.topology.owners,
            "ports": {
                switch: sorted(ports) for switch, ports in self.port_states.items()
            },
            "resources": dict(self.topology.resources),
            "support_switches": {
                slot: row["support_switches"]
                for slot, row in self.network.controllers.items()
            },
        }

    def _pull_peers(self) -> None:
        """Collect peer presence while discarding failed or expired responses."""
        timeout = min(self.network.telemetry["peer_poll_seconds"], 0.5)
        for peer, item in self.network.controllers.items():
            if peer == self.controller_id:
                continue
            url = "http://{}:{}/network/state".format(item["host"], item["http_port"])
            try:
                with self.http.open(url, timeout=timeout) as response:
                    frame = json.load(response)
                if (
                    frame["controller_id"] != peer
                    or frame["network"] != self.network.fingerprint
                ):
                    continue
                cursor = (frame["incarnation"], frame["sequence"])
                if peer not in self.peers or cursor >= self.peers[peer][0]:
                    self.peers[peer] = (cursor, time.monotonic(), frame)
                    self.seen_peers.add(peer)
            except (URLError, OSError, ValueError, KeyError) as error:
                self.logger.debug("Peer %s unavailable: %s", peer, error)

    def _run(self) -> None:
        """Interleave peer reconciliation and telemetry polls until the controller stops."""
        while True:
            try:
                self._pull_peers()
                with self.lock:
                    self._reconcile()
            except Exception:
                self.logger.exception("Unable to reconcile EAGLE topology")
                with self.lock:
                    self.topology_key = None
                    self.telemetry.reset()
            hub.sleep(self.network.telemetry["peer_poll_seconds"])

    def _reconcile(self) -> None:
        """Merge peer actions, elect owners, and reconcile OpenFlow rules after topology changes."""
        now = time.monotonic()
        expiry = self.network.telemetry["peer_timeout_seconds"]
        frames = {
            peer: frame
            for peer, (_, received, frame) in self.peers.items()
            if now - received <= expiry
        }
        for frame in frames.values():
            self.topology.merge_resources(frame["resources"])
            self.seen_switches.update(frame["ports"])
        frames[self.controller_id] = self.presence()
        if (
            self.seen_peers != self.network.controllers.keys()
            or self.seen_switches != self.network.switches.keys()
        ):
            return
        self.topology.update_presence(frames)
        self.sequence += 1
        desired_datapaths = {
            node: datapath
            for node, datapath in self.datapaths.items()
            if self.topology.owners[self.network.switches[node]["controller"]]
            == self.controller_id
        }
        for node, datapath in sorted(self.datapaths.items()):
            is_owner = node in desired_datapaths
            role, requested_at = self.requested_roles.get(node, (False, 0.0))
            if is_owner and (
                not role or (node not in self.master and now - requested_at >= expiry)
            ):
                self.requested_roles[node] = (True, now)
                request_master(datapath, time.time_ns())
            elif role and not is_owner:
                self.requested_roles[node] = (False, now)
                self.master.discard(node)
                self.installed.pop(node, None)
                request_slave(datapath, time.time_ns())
        if desired_datapaths.keys() - self.master:
            self.topology_key = None
            self.telemetry.reset()
            return

        graph = self.topology.active
        key = (
            frozenset(graph.nodes),
            frozenset(frozenset(edge) for edge in graph.edges),
            tuple(self.topology.owners.items()),
            frozenset(desired_datapaths),
            tuple(tuple(row["revision"]) for row in self.topology.resources.values()),
        )
        if key != self.topology_key or (
            self.pending_barriers and now >= self.barrier_deadline
        ):
            self.generation += 1
            self.telemetry.reset()
            rules = plan_forwarding_rules(graph, self.network)
            self._dispatch_rules(desired_datapaths, rules, now)
            self.topology_key = key
            self.route_error = None
            self.next_poll = now
        if not self.pending_barriers and now >= self.next_poll:
            self.telemetry.request_round(desired_datapaths, self.generation)
            self.next_poll = max(
                now,
                self.next_poll + self.network.telemetry["poll_seconds"],
            )

    def _dispatch_rules(
        self,
        datapaths: Dict[str, Datapath],
        rules: Dict[str, ForwardingRules],
        now: float,
    ) -> None:
        """Send rule updates and a barrier while tracking errors by transaction ID."""
        started = time.monotonic()
        previous_pending = self.pending_barriers
        self.pending_barriers = {}
        for switch, datapath in sorted(datapaths.items()):
            old = None if switch in previous_pending else self.installed.get(switch)
            desired = rules[switch]
            if old == desired:
                continue
            apply_forwarding_rules(datapath, old, desired)
            barrier = datapath.ofproto_parser.OFPBarrierRequest(datapath)
            datapath.set_xid(barrier)
            self.pending_barriers[switch] = (datapath, barrier.xid)
            datapath.send_msg(barrier)
            self.installed[switch] = desired
        self.barrier_deadline = now + self.network.telemetry["peer_timeout_seconds"]
        self.rule_timing = {
            "dispatch_seconds": time.monotonic() - started,
            "dispatched_at": time.time(),
        }

    def stats(self, slot: str) -> Optional[Dict[str, Any]]:  # noqa: C901
        """Publish a coherent slot snapshot from active topology and a completed poll round."""
        if (
            self.topology.owners.get(slot) != self.controller_id
            or self.telemetry.latest is None
        ):
            return None
        sample = self.telemetry.latest
        duration = sample["sample_duration_seconds"]
        graph = self.topology.observation_graph(slot)
        weights = {}
        switch_rates = {}
        active = self.topology.active
        for node, ports in sample["ports"].items():
            if self.network.switches[node]["controller"] != slot:
                continue
            for port, (rx, tx) in ports.items():
                neighbor = self.network.ports[node][port]
                if active.has_edge(node, neighbor):
                    switch_rates[node] = switch_rates.get(node, 0.0) + tx * 8 / duration
                if not graph.has_edge(node, neighbor):
                    continue
                capacity = graph[node][neighbor]["capacity_bps"]
                weights.setdefault(node, {})[neighbor] = max(
                    weights.get(node, {}).get(neighbor, 0),
                    compute_link_utilization(tx, duration, capacity),
                )
                weights.setdefault(neighbor, {})[node] = max(
                    weights.get(neighbor, {}).get(node, 0),
                    compute_link_utilization(rx, duration, capacity),
                )
        delivered = {}
        for node, flows in sorted(sample["flows"].items()):
            if self.network.switches[node]["controller"] != slot:
                continue
            for (source, target), count in flows.items():
                if target in self.network.graph[node]:
                    delivered[source + " " + target] = count
        active_switches, active_ports = self.topology.resource_usage(slot)
        links = []
        for source, target in graph.edges:
            weight = max(
                weights.get(source, {}).get(target, 0),
                weights.get(target, {}).get(source, 0),
            )
            links.append({"source": source, "target": target, "weight": weight})
        return {
            "schema_version": 1,
            "network": self.network.fingerprint,
            "controller_id": slot,
            "served_by": self.controller_id,
            "incarnation": self.incarnation,
            "sample_id": sample["sample_id"],
            "sampled_at": sample["sampled_at"],
            "window_start": sample["window_start"],
            "sample_duration_seconds": duration,
            "delivered_bytes": delivered,
            "topology": {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    dict(attributes, id=node)
                    for node, attributes in graph.nodes(data=True)
                ],
                "links": links,
            },
            "resources": dict(self.topology.resources),
            "usage": {
                "active_switches": active_switches,
                "active_ports": active_ports,
                "switch_egress_bps": switch_rates,
                "links": sorted(sorted(edge) for edge in graph.edges),
            },
            "port_counters": {
                node: counters
                for node, counters in sample["port_counters"].items()
                if self.network.switches[node]["controller"] == slot
            },
            "flow_counters": {
                node: counters
                for node, counters in sample["flow_counters"].items()
                if self.network.switches[node]["controller"] == slot
            },
            "rule_timing": dict(self.rule_timing),
        }
