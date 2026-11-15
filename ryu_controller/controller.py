"""OpenFlow 1.3 routing, peer state exchange, and the EAGLE HTTP interface."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from webob import Request, Response

from ryu_controller.openflow_rules import (
    apply_forwarding_rules,
    request_master,
    request_slave,
)
from ryu_controller.routing import plan_forwarding_rules
from ryu_controller.telemetry import Telemetry, compute_link_utilization
from ryu_controller.topology_state import TopologyState, load_scenario

if TYPE_CHECKING:
    from ryu.controller.controller import Datapath
    from ryu.ofproto.ofproto_v1_3_parser import OFPPort

    from .routing import ForwardingRules


class EagleController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scenario = load_scenario(os.environ["EAGLE_SCENARIO"])
        self.controller_id = os.environ["EAGLE_CONTROLLER_ID"]
        self.incarnation = time.time_ns()
        self.sequence = 0
        self.topology = TopologyState(self.scenario)
        self.telemetry = Telemetry(self.scenario)
        self.switch_for_dpid = {
            row["dpid"]: node for node, row in self.scenario.switches.items()
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
        datapath = event.datapath
        switch = self.switch_for_dpid.get(datapath.id)
        if switch is None:
            return
        with self.lock:
            if event.state == MAIN_DISPATCHER:
                expected = self.scenario.ports[switch].keys()
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
                    if _port_is_up(datapath, datapath.ports[port])
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
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        port = message.desc.port_no
        with self.lock:
            if (
                self.datapaths.get(switch) is not message.datapath
                or port not in self.scenario.ports[switch]
            ):
                return
            if (
                message.reason == message.datapath.ofproto.OFPPR_DELETE
                or not _port_is_up(message.datapath, message.desc)
            ):
                self.port_states[switch].discard(port)
            else:
                self.port_states[switch].add(port)

    @set_ev_cls(ofp_event.EventOFPRoleReply, MAIN_DISPATCHER)
    def on_role_reply(self, event: ofp_event.EventOFPRoleReply) -> None:
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
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        with self.lock:
            if self.pending_barriers.get(switch) == (message.datapath, message.xid):
                del self.pending_barriers[switch]
                if not self.pending_barriers:
                    self.rule_timing["barrier_at"] = time.time()

    @set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
    def on_openflow_error(self, event: ofp_event.EventOFPErrorMsg) -> None:
        message = event.msg
        switch = self.switch_for_dpid.get(message.datapath.id)
        self.logger.warning(
            "OpenFlow error on %s: type=%s code=%s", switch, message.type, message.code
        )
        with self.lock:
            if self.datapaths.get(switch) is message.datapath and (
                switch in self.master or self.requested_roles.get(switch, (False, 0))[0]
            ):
                self._forget_programming(switch)
                self.route_error = "OpenFlow programming rejected on " + switch

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def on_stats(
        self,
        event: ofp_event.EventOFPPortStatsReply | ofp_event.EventOFPFlowStatsReply,
    ) -> None:
        with self.lock:
            self.telemetry.receive(event.msg)

    def presence(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scenario": self.scenario.fingerprint,
            "controller_id": self.controller_id,
            "incarnation": self.incarnation,
            "sequence": self.sequence,
            "owners": self.topology.owners,
            "ports": {
                switch: sorted(ports) for switch, ports in self.port_states.items()
            },
            "resources": dict(self.topology.resources),
        }

    def _pull_peers(self) -> None:
        timeout = min(self.scenario.telemetry["peer_poll_seconds"], 0.5)
        for peer, item in self.scenario.controllers.items():
            if peer == self.controller_id:
                continue
            url = "http://{}:{}/network/state".format(item["host"], item["http_port"])
            try:
                with self.http.open(url, timeout=timeout) as response:
                    frame = json.load(response)
                if (
                    frame["controller_id"] != peer
                    or frame["scenario"] != self.scenario.fingerprint
                ):
                    continue
                cursor = (frame["incarnation"], frame["sequence"])
                if peer not in self.peers or cursor >= self.peers[peer][0]:
                    self.peers[peer] = (cursor, time.monotonic(), frame)
                    self.seen_peers.add(peer)
            except (URLError, OSError, ValueError, KeyError) as error:
                self.logger.debug("Peer %s unavailable: %s", peer, error)

    def _run(self) -> None:
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
            hub.sleep(self.scenario.telemetry["peer_poll_seconds"])

    def _reconcile(self) -> None:
        now = time.monotonic()
        expiry = self.scenario.telemetry["peer_timeout_seconds"]
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
            self.seen_peers != self.scenario.controllers.keys()
            or self.seen_switches != self.scenario.switches.keys()
        ):
            return
        self.topology.update_presence(frames)
        self.sequence += 1
        desired_datapaths = {
            node: datapath
            for node, datapath in self.datapaths.items()
            if self.topology.owners[self.scenario.switches[node]["controller"]]
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
            rules = plan_forwarding_rules(graph, self.scenario)
            self._dispatch_rules(desired_datapaths, rules, now)
            self.topology_key = key
            self.route_error = None
            self.next_poll = now
        if not self.pending_barriers and now >= self.next_poll:
            self.telemetry.request_round(desired_datapaths, self.generation)
            if not desired_datapaths:
                # An elected controller with no present switches publishes an empty window.
                self.telemetry._publish(self.telemetry.pending)
                self.telemetry.pending = None
            self.next_poll = max(
                now, self.next_poll + self.scenario.telemetry["poll_seconds"]
            )

    def _dispatch_rules(
        self,
        datapaths: dict[str, Datapath],
        rules: dict[str, ForwardingRules],
        now: float,
    ) -> None:
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
        self.barrier_deadline = now + self.scenario.telemetry["peer_timeout_seconds"]
        self.rule_timing = {
            "dispatch_seconds": time.monotonic() - started,
            "dispatched_at": time.time(),
        }

    def stats(self, slot: str, view: str) -> dict[str, Any] | None:
        if (
            self.topology.owners.get(slot) != self.controller_id
            or self.telemetry.latest is None
        ):
            return None
        sample = self.telemetry.latest
        duration = sample["sample_duration_seconds"]
        # The checkout divided raw deltas by a Mi-based denominator with no elapsed-time term.
        utilization_duration = (
            duration if view == "active" else (1024 * 1024) / 1_000_000
        )
        graph = self.topology.observation_graph(slot, view)
        weights = {}
        switch_rates = {}
        active = self.topology.active
        for node, ports in sample["ports"].items():
            if self.scenario.switches[node]["controller"] != slot:
                continue
            for port, (rx, tx) in ports.items():
                neighbor = self.scenario.ports[node][port]
                if active.has_edge(node, neighbor):
                    switch_rates[node] = switch_rates.get(node, 0.0) + tx * 8 / duration
                if not graph.has_edge(node, neighbor):
                    continue
                capacity = graph[node][neighbor]["capacity_bps"]
                weights.setdefault(node, {})[neighbor] = max(
                    weights.get(node, {}).get(neighbor, 0),
                    compute_link_utilization(tx, utilization_duration, capacity),
                )
                weights.setdefault(neighbor, {})[node] = max(
                    weights.get(neighbor, {}).get(node, 0),
                    compute_link_utilization(rx, utilization_duration, capacity),
                )
        delivered = {}
        legacy = {}
        for node, flows in sorted(sample["flows"].items()):
            if self.scenario.switches[node]["controller"] != slot:
                continue
            for (source, target), count in flows.items():
                flow = source + " " + target
                legacy.setdefault(flow, count)
                if target in self.scenario.graph[node]:
                    delivered[flow] = count
                    legacy[flow] = count
        active_switches, active_ports = self.topology.resource_usage(slot)
        links = []
        for source, target in graph.edges:
            weight = max(
                weights.get(source, {}).get(target, 0),
                weights.get(target, {}).get(source, 0),
            )
            # The legacy encoder saw unset weights on edges with no samples.
            row = {"source": source, "target": target}
            if (
                view == "active"
                or target in weights.get(source, {})
                or source in weights.get(target, {})
            ):
                row["weight"] = weight
            links.append(row)
        return {
            "schema_version": 1,
            "scenario": self.scenario.fingerprint,
            "controller_id": slot,
            "served_by": self.controller_id,
            "incarnation": self.incarnation,
            "sample_id": sample["sample_id"],
            "sampled_at": sample["sampled_at"],
            "window_start": sample["window_start"],
            "sample_duration_seconds": duration,
            "stats": {"data": weights},
            "throughput": legacy if view == "configured" else delivered,
            "demands": legacy,
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
                "links": sorted(
                    sorted(edge)
                    for edge in self.topology.observation_graph(slot, "active").edges
                ),
            },
            "port_counters": {
                node: counters
                for node, counters in sample["port_counters"].items()
                if self.scenario.switches[node]["controller"] == slot
            },
            "flow_counters": {
                node: counters
                for node, counters in sample["flow_counters"].items()
                if self.scenario.switches[node]["controller"] == slot
            },
            "rule_timing": dict(self.rule_timing),
        }


def _port_is_up(datapath: Datapath, port: OFPPort) -> bool:
    ofp = datapath.ofproto
    return not (port.config & ofp.OFPPC_PORT_DOWN or port.state & ofp.OFPPS_LINK_DOWN)


def _response(data: object, status: int = 200) -> Response:
    return Response(
        status=status, content_type="application/json", body=json.dumps(data).encode()
    )


class EagleApi(ControllerBase):
    def __init__(
        self,
        request: Request,
        link: Any,
        data: dict[str, EagleController],
        **config: Any,
    ) -> None:
        super().__init__(request, link, data, **config)
        self.app = data["app"]

    @route("eagle", "/network/state", methods=["GET"])
    def get_state(self, request: Request, **kwargs: Any) -> Response:
        with self.app.lock:
            return _response(self.app.presence())

    @route("eagle", "/network/stats", methods=["GET"])
    def get_stats(self, request: Request, **kwargs: Any) -> Response:
        slot = request.GET.get("controller_id", self.app.controller_id)
        view = request.GET.get("view", "configured")
        if slot not in self.app.scenario.controllers or view not in (
            "configured",
            "active",
        ):
            return _response(
                {"error": "Unknown controller slot or observation view"}, 400
            )
        with self.app.lock:
            snapshot = self.app.stats(slot, view)
            if snapshot is None:
                return _response(
                    {
                        "error": self.app.route_error
                        or "Waiting for ownership and a complete sample",
                        "owner": self.app.topology.owners.get(slot),
                    },
                    503,
                )
            return _response(snapshot)

    @route("eagle", "/model/actions", methods=["PUT"])
    def put_actions(self, request: Request, **kwargs: Any) -> Response:
        slot = request.GET.get("controller_id", self.app.controller_id)
        if (
            request.GET.get("scenario", self.app.scenario.fingerprint)
            != self.app.scenario.fingerprint
        ):
            return _response({"error": "Scenario fingerprint mismatch"}, 409)
        try:
            payload = request.json
        except ValueError:
            payload = None
        if slot not in self.app.scenario.controllers:
            return _response({"error": "Unknown controller slot"}, 400)
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or type(payload[0]) is not int
            or payload[0] not in (0, 1)
        ):
            return _response({"error": "Expected [0] or [1]"}, 400)
        with self.app.lock:
            if self.app.topology.owners.get(slot) != self.app.controller_id:
                return _response(
                    {
                        "error": "Controller does not own this slot",
                        "owner": self.app.topology.owners.get(slot),
                    },
                    503,
                )
            current = self.app.topology.resources[slot]
            revision = [
                max(time.time_ns(), current["revision"][0] + 1),
                self.app.controller_id,
            ]
            if current["action"] != payload[0]:
                self.app.topology.set_support_action(slot, payload[0], revision)
                self.app.telemetry.reset()
                self.app.sequence += 1
            update = self.app.topology.resources[slot]
            accepted_at = time.time()
            return (
                _response(payload, 200)
                if request.GET.get("details") != "1"
                else _response(
                    {
                        "controller_id": slot,
                        "served_by": self.app.controller_id,
                        "action": update["action"],
                        "scenario": self.app.scenario.fingerprint,
                        "revision": update["revision"],
                        "accepted_at": accepted_at,
                        "changed": current != update,
                    }
                )
            )
