import argparse
import hashlib
import ipaddress
import json
import os
import re
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from mininet.link import Link
from mininet.net import Mininet
from mininet.node import RemoteController

if __package__:
    from .traffic import TrialTraffic
else:
    from traffic import TrialTraffic


ROOT = Path(__file__).resolve().parents[1]


def validate_scenario(scenario: dict[str, Any]) -> None:
    """Check values passed to Mininet's system commands before creating nodes."""
    if scenario.get("schema_version") != 1:
        raise ValueError("Expected scenario schema version 1")
    names = set()
    for group in ("controllers", "switches", "hosts"):
        for row in scenario[group]:
            name = row["id"]
            if name in names or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid or duplicate Mininet node name: {name!r}")
            names.add(name)
    for controller in scenario["controllers"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", controller["host"]):
            raise ValueError("Use an IPv4 address or host name for Mininet controllers")
        if (
            type(controller["openflow_port"]) is not int
            or not 0 < controller["openflow_port"] < 65536
        ):
            raise ValueError("Invalid OpenFlow listen port")
    dpids = set()
    ports = {row["id"]: set() for row in scenario["switches"] + scenario["hosts"]}
    for switch in scenario["switches"]:
        dpid = switch["dpid"]
        if type(dpid) is not int or not 0 < dpid < (1 << 64) or dpid in dpids:
            raise ValueError("Invalid or duplicate datapath ID for " + switch["id"])
        dpids.add(dpid)
    for host in scenario["hosts"]:
        ipaddress.IPv4Interface(host["ip"])
        if not re.fullmatch(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", host["mac"]):
            raise ValueError("Invalid MAC address for " + host["id"])
    for link in scenario["links"]:
        for endpoint in ("source", "target"):
            node, port = link[endpoint], link[endpoint + "_port"]
            if (
                node not in ports
                or type(port) is not int
                or port < 0
                or len(f"{node}-eth{port}") > 15
                or port in ports[node]
            ):
                raise ValueError("Invalid or duplicate Mininet interface for " + node)
            ports[node].add(port)
        if type(link["capacity_bps"]) is not int or link["capacity_bps"] <= 0:
            raise ValueError("Link capacity must be positive integer bps")


def build_network(
    network: Mininet,
    scenario: dict[str, Any],
    remote_controller: type[RemoteController],
) -> dict[frozenset[str], Link]:
    """Create switches and host links with the configured OpenFlow port numbers."""
    for controller in scenario["controllers"]:
        network.addController(
            controller["id"],
            controller=remote_controller,
            ip=controller["host"],
            port=controller["openflow_port"],
        )
    for switch in scenario["switches"]:
        network.addSwitch(
            switch["id"],
            dpid="{:016x}".format(switch["dpid"]),
            protocols="OpenFlow13,OpenFlow15",
            failMode="secure",
            batch=False,
        )
    for host in scenario["hosts"]:
        network.addHost(host["id"], ip=host["ip"], mac=host["mac"])
    links = {}
    for edge in scenario["links"]:
        source, target = edge["source"], edge["target"]
        key = frozenset((source, target))
        if source == target or key in links or edge["capacity_bps"] <= 0:
            raise ValueError(f"Invalid or repeated link {source}–{target}")
        link = network.addLink(
            network[source],
            network[target],
            port1=edge["source_port"],
            port2=edge["target_port"],
            bw=edge["capacity_bps"] / 1_000_000,
        )
        links[key] = link
    return links


class EmulationSession:
    def __init__(
        self,
        network: Mininet,
        scenario: dict[str, Any],
        fingerprint: str,
        links: dict[frozenset[str], Link],
        traffic: TrialTraffic,
    ) -> None:
        self.network = network
        self.fingerprint = fingerprint
        self.links = links
        self.traffic = traffic
        self.supports = {
            row["id"]: row["support_switch"] for row in scenario["controllers"]
        }
        self.traffic_traces = tuple(scenario.get("traffic", {}))
        self.faults = set()
        self.disabled = set()
        self.down_links = set()
        self.mode = "forwarding"

    def _set_blocked_links(self) -> list[dict[str, Any]]:
        blocked = set(self.faults)
        if self.mode == "interfaces":
            blocked.update(link for link in self.links if link & self.disabled)
        changes = []
        for key in sorted(
            self.down_links ^ blocked, key=lambda endpoints: tuple(sorted(endpoints))
        ):
            status = "down" if key in blocked else "up"
            link = self.links[key]
            start = time.monotonic()
            for interface in (link.intf1, link.intf2):
                output = interface.ifconfig(status)
                if output:
                    raise RuntimeError(
                        f"Unable to set {interface} {status}: {output.strip()}"
                    )
            if status == "down":
                self.down_links.add(key)
            else:
                self.down_links.discard(key)
            changes.append(
                {
                    "link": sorted(key),
                    "enabled": status == "up",
                    "seconds": time.monotonic() - start,
                }
            )
        self.down_links = blocked
        return changes

    def _set_ovs_state(self, switch: str, enabled: bool) -> None:
        if enabled:
            self.network[switch].start(self.network.controllers)
        else:
            subprocess.run(["ovs-vsctl", "--if-exists", "del-br", switch], check=True)

    def set_resources(self, actions: dict[str, int]) -> dict[str, Any]:
        """Apply optional physical changes for configured supporting switches only."""
        desired = set()
        for slot, action in actions.items():
            if (
                slot not in self.supports
                or type(action) is not int
                or action not in (0, 1)
            ):
                raise ValueError("Expected a binary action for a configured slot")
            if not action:
                desired.add(self.supports[slot])
        if set(actions) != self.supports.keys():
            raise ValueError("Supply exactly one resource action per controller")
        timings = []
        for switch in sorted(self.disabled ^ desired):
            started = time.monotonic()
            if self.mode == "ovs":
                self._set_ovs_state(switch, switch not in desired)
                if switch in desired:
                    self.disabled.add(switch)
                else:
                    self.disabled.discard(switch)
            timings.append(
                {
                    "switch": switch,
                    "enabled": switch not in desired,
                    "ovs_seconds": time.monotonic() - started
                    if self.mode == "ovs"
                    else None,
                }
            )
        self.disabled = desired
        start = time.monotonic()
        link_changes = self._set_blocked_links()
        return {
            "mode": self.mode,
            "changes": timings,
            "interface_seconds": time.monotonic() - start
            if self.mode == "interfaces"
            else None,
            "interface_changes": link_changes,
            "applied_at": time.time(),
        }

    def command(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "status":
            return {
                "scenario": self.fingerprint,
                "traffic_profiles": self.traffic_traces,
            }
        if operation == "finish":
            return self.traffic.finish()
        if operation == "reset":
            mode = request.get("resource_mode", "forwarding")
            if mode not in ("forwarding", "interfaces", "ovs"):
                raise ValueError("Unknown resource mode")
            self.traffic.finish()
            self.traffic.records = []
            self.faults.clear()
            self.set_resources({slot: 1 for slot in self.supports})
            self.mode = mode
            return {"scenario": self.fingerprint}
        if operation == "traffic":
            return self.traffic.start(request["profile"])
        if operation == "resources":
            return self.set_resources(request["actions"])
        if operation == "link":
            key = frozenset((request["source"], request["target"]))
            if key not in self.links or type(request["up"]) is not bool:
                raise ValueError("Expected a configured link and a boolean up value")
            if request["up"]:
                self.faults.discard(key)
            else:
                self.faults.add(key)
            self._set_blocked_links()
            return {"applied_at": time.time(), "up": request["up"]}
        raise ValueError("Unknown emulation operation")


class ControlRequest(socketserver.StreamRequestHandler):
    server: Any

    def handle(self) -> None:
        self.connection.settimeout(10)
        credentials = self.connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, 12
        )
        _, uid, _ = struct.unpack("3i", credentials)
        if uid not in (0, self.server.owner_uid):
            return
        try:
            request = json.loads(self.rfile.readline(65536))
            if not isinstance(request, dict):
                raise TypeError("Expected a JSON object")
            response = {"result": self.server.session.command(request)}
        except (
            OSError,
            TypeError,
            KeyError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            response = {"error": str(error)}
        try:
            self.wfile.write(json.dumps(response).encode() + b"\n")
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios/v0.json")
    parser.add_argument(
        "--cli", action="store_true", help="Open an interactive Mininet CLI"
    )
    args = parser.parse_args()
    if os.geteuid() != 0 or sys.prefix != sys.base_prefix:
        parser.error("Use sudo python3 with the system Mininet installation")
    try:
        from mininet.cli import CLI
        from mininet.link import TCLink
        from mininet.net import Mininet
        from mininet.node import OVSKernelSwitch, RemoteController
    except ImportError as error:
        parser.error(f"System Python must be able to import Mininet: {error}")
    raw = args.scenario.read_bytes()
    scenario = json.loads(raw)
    try:
        validate_scenario(scenario)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    fingerprint = hashlib.sha256(raw).hexdigest()
    owner_uid = int(os.environ.get("SUDO_UID", "0"))
    owner_gid = int(os.environ.get("SUDO_GID", "0"))
    network = Mininet(
        topo=None, controller=None, switch=OVSKernelSwitch, link=TCLink, build=False
    )

    def stop_on_signal(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    with tempfile.TemporaryDirectory(prefix="eagle-mininet-") as directory:
        session_dir = Path(directory)
        session_dir.chmod(0o711)
        traffic = TrialTraffic(
            network,
            scenario.get("traffic", {}),
            args.scenario.resolve().parent,
            session_dir,
            owner_uid,
        )
        try:
            links = build_network(network, scenario, RemoteController)
            network.start()
            network.staticArp()
            address = str(session_dir / "control.sock")
            with socketserver.UnixStreamServer(address, ControlRequest) as server:
                server.owner_uid = owner_uid
                server.session = EmulationSession(
                    network, scenario, fingerprint, links, traffic
                )
                os.chown(address, owner_uid, owner_gid)
                os.chmod(address, 0o600)
                socket_path = Path(__file__).resolve().with_name(".socket")
                address_file = socket_path.open("x", encoding="utf-8")
                try:
                    with address_file:
                        address_file.write(address)
                    if not args.cli:
                        server.serve_forever(poll_interval=0.25)
                    else:
                        thread = threading.Thread(target=server.serve_forever, daemon=True)
                        thread.start()
                        try:
                            CLI(network)
                        finally:
                            server.shutdown()
                            thread.join()
                finally:
                    socket_path.unlink(missing_ok=True)
        except KeyboardInterrupt:
            pass
        finally:
            traffic.finish()
            network.stop()


if __name__ == "__main__":
    main()
