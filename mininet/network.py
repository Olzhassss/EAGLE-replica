"""Validate command-facing identifiers and populate a generated Mininet network."""

import ipaddress
import re
from typing import Any

from network_schema import validate_support_layout


def validate_network(network: dict[str, Any]) -> None:
    """Check values passed to Mininet's system commands before creating nodes."""
    validate_support_layout(network)
    names = set()
    for group in ("controllers", "switches", "hosts"):
        for row in network[group]:
            name = row["id"]
            if name in names or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid or duplicate Mininet node name: {name!r}")
            names.add(name)
    for controller in network["controllers"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", controller["host"]):
            raise ValueError("Use an IPv4 address or host name for Mininet controllers")
        if (
            type(controller["openflow_port"]) is not int
            or not 0 < controller["openflow_port"] < 65536
        ):
            raise ValueError("Invalid OpenFlow listen port")
    dpids = set()
    ports = {row["id"]: set() for row in network["switches"] + network["hosts"]}
    for switch in network["switches"]:
        dpid = switch["dpid"]
        if type(dpid) is not int or not 0 < dpid < (1 << 64) or dpid in dpids:
            raise ValueError("Invalid or duplicate datapath ID for " + switch["id"])
        dpids.add(dpid)
    for host in network["hosts"]:
        ipaddress.IPv4Interface(host["ip"])
        if not re.fullmatch(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", host["mac"]):
            raise ValueError("Invalid MAC address for " + host["id"])
    for link in network["links"]:
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


def populate_network(
    mininet: Any,
    network: dict[str, Any],
    remote_controller: Any,
) -> None:
    """Add configured controllers, switches, hosts, and links to a Mininet instance."""
    for controller in network["controllers"]:
        mininet.addController(
            controller["id"],
            controller=remote_controller,
            ip=controller["host"],
            port=controller["openflow_port"],
        )
    for switch in network["switches"]:
        mininet.addSwitch(
            switch["id"],
            dpid="{:016x}".format(switch["dpid"]),
            protocols="OpenFlow13,OpenFlow15",
            failMode="secure",
            batch=False,
        )
    for host in network["hosts"]:
        mininet.addHost(host["id"], ip=host["ip"], mac=host["mac"])
    for edge in network["links"]:
        source, target = edge["source"], edge["target"]
        mininet.addLink(
            mininet[source],
            mininet[target],
            port1=edge["source_port"],
            port2=edge["target_port"],
            bw=edge["capacity_bps"] / 1_000_000,
        )
