"""Host addressing and dataset host metadata."""

import json
import re
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Interface
from pathlib import Path


@dataclass(frozen=True)
class Host:
    """A traffic endpoint with packed MAC and IPv4 addresses for packet rewriting."""

    name: str
    mac: bytes
    ip: bytes

    def to_dict(self) -> dict:
        return {
            "id": self.name,
            "mac": self.mac.hex(":"),
            "ip": f"{IPv4Address(self.ip)}/8",
        }


def hosts_for(count: int) -> list[Host]:
    """Create deterministic traffic endpoints with unique MAC and IPv4 addresses."""
    return [
        Host(
            f"h{index}",
            b"\x02\x00" + index.to_bytes(4, "big"),
            IPv4Address(0x0A000000 + index).packed,
        )
        for index in range(1, count + 1)
    ]


def dataset_hosts(path: Path) -> list[dict]:
    """Validate explicit host metadata from a prepared dataset manifest."""
    rows = json.loads(path.read_text()).get("hosts")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Regenerate {path} to include explicit host metadata")
    hosts = [
        dict(row, mac=row["mac"].lower(), ip=str(IPv4Interface(row["ip"])))
        for row in rows
    ]
    if (
        any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", host["id"])
            or not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", host["mac"])
            for host in hosts
        )
        or any(
            len({host[field] for host in hosts}) != len(hosts) for field in ("id", "mac")
        )
        or len({IPv4Interface(host["ip"]).ip for host in hosts}) != len(hosts)
    ):
        raise ValueError("Hosts need valid and unique IDs, MACs and IPv4 addresses")
    return hosts
