"""Launch a generated network and serve prepared traffic jobs."""

from __future__ import annotations

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
from pathlib import Path
from types import FrameType
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
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
    def __init__(self, fingerprint: str, traffic: TrialTraffic) -> None:
        self.fingerprint = fingerprint
        self.traffic = traffic

    def command(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "status":
            return {"scenario": self.fingerprint}
        if operation == "finish":
            return self.traffic.finish(request.get("drain_seconds", 0))
        if operation == "reset":
            self.traffic.finish()
            return {"scenario": self.fingerprint}
        if operation == "traffic":
            return self.traffic.start(
                request["jobs"], request["duration_seconds"],
                measure_flows=request.get("measure_flows", True),
            )
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
            request = json.loads(self.rfile.readline(64 * 1024 * 1024))
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
    parser.add_argument("--scenario", type=Path, default=ROOT / "networks/v0.json")
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
        traffic = TrialTraffic(network, session_dir, owner_uid)
        try:
            build_network(network, scenario, RemoteController)
            network.start()
            network.staticArp()
            address = str(session_dir / "control.sock")
            with socketserver.UnixStreamServer(address, ControlRequest) as server:
                server.owner_uid = owner_uid
                server.session = EmulationSession(fingerprint, traffic)
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
