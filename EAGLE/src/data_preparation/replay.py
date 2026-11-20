"""Write timestamped, address-remapped Ethernet captures for each host direction."""

import json
import math
import re
import struct
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Interface
from pathlib import Path

from scapy.utils import checksum

from .analyze_trace import capture_packets, ethernet_payload
from .load_and_split import PCAP_HEADER, SECOND, Window, write_packet
from .traffic_loads import write_chart

TRANSPORT_CHECKSUM = {6: 16, 17: 6, 58: 2}


@dataclass(frozen=True)
class Host:
    name: str
    mac: bytes
    ip: bytes

    def to_dict(self) -> dict:
        return {
            "id": self.name,
            "mac": self.mac.hex(":"),
            "ip": f"{IPv4Address(self.ip)}/8",
        }


def hosts_for(count: int, manifest: Path | None = None) -> list[Host]:
    if manifest is None:
        return [
            Host(
                f"h{index}",
                b"\x02\x00" + index.to_bytes(4, "big"),
                IPv4Address(0x0A000000 + index).packed,
            )
            for index in range(1, count + 1)
        ]
    rows = json.loads(manifest.read_text())["hosts"]
    if len(rows) != count:
        raise ValueError(f"{manifest} must define exactly {count} hosts")
    hosts = [
        Host(
            row["id"],
            bytes.fromhex(row["mac"].replace(":", "")),
            IPv4Interface(row["ip"]).ip.packed,
        )
        for row in rows
    ]
    if any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", host.name) or len(host.mac) != 6
        for host in hosts
    ) or any(
        len({getattr(host, field) for host in hosts}) != count
        for field in ("name", "mac", "ip")
    ):
        raise ValueError("Hosts need valid and unique IDs, MACs and IPv4 addresses")
    return hosts


def dataset_hosts(path: Path) -> list[dict]:
    """Read host metadata, including manifests generated before it was explicit."""
    data = json.loads(path.read_text())
    if "hosts" in data:
        return [host.to_dict() for host in hosts_for(len(data["hosts"]), path)]
    rows = data.get("series", data.get("captures", []))
    names = {row[endpoint] for row in rows for endpoint in ("source", "target")}
    if not names:
        raise ValueError(f"No host information in {path}")
    capture_path = path
    if "captures" not in data:
        capture_path = path.with_name(path.stem.split("_")[0] + "_pcap.json")
    addresses = {}
    if capture_path.exists():
        captures = json.loads(capture_path.read_text()).get("captures", [])
        for row in captures:
            for _, _, frame in capture_packets(capture_path.parent / row["path"]):
                ethertype, offset = ethernet_payload(frame)
                if ethertype != b"\x08\x00" or len(frame) < offset + 20:
                    continue
                for name, mac, ip in (
                    (row["source"], frame[6:12], frame[offset + 12 : offset + 16]),
                    (row["target"], frame[:6], frame[offset + 16 : offset + 20]),
                ):
                    host = Host(name, mac, ip).to_dict()
                    if name in addresses and addresses[name] != host:
                        raise ValueError(
                            f"Inconsistent addresses for {name} in {capture_path}"
                        )
                    addresses[name] = host
                break
    result = []
    for name in sorted(names, key=lambda value: (len(value), value)):
        if name in addresses:
            result.append(addresses[name])
        elif re.fullmatch(r"h[1-9][0-9]*", name):
            index = int(name[1:])
            result.append(
                Host(
                    name,
                    b"\x02\x00" + index.to_bytes(4, "big"),
                    IPv4Address(0x0A000000 + index).packed,
                ).to_dict()
            )
        else:
            raise ValueError(f"Regenerate {path} to include explicit host metadata")
    return result


def _transport_checksum(
    frame: bytearray,
    start: int,
    end: int,
    protocol: int,
    old: bytes,
    addresses: bytes,
    partial: bool,
) -> None:
    checksum_offset = TRANSPORT_CHECKSUM.get(protocol)
    if checksum_offset is None or start + checksum_offset + 2 > min(end, len(frame)):
        return
    ipv4 = len(addresses) == 8
    field = start + checksum_offset
    previous = struct.unpack_from("!H", frame, field)[0]
    if ipv4 and protocol == 17 and previous == 0:
        return  # IPv4 UDP allows a disabled checksum.
    if partial or end > len(frame):
        # The packet is fragmented or truncated. Adjust the old pseudoheader sum.
        updated = checksum(
            struct.pack("!H", previous ^ 0xFFFF)
            + bytes(value ^ 0xFF for value in old)
            + addresses
        )
    else:
        frame[field : field + 2] = b"\0\0"
        trailer = (
            struct.pack("!BBH", 0, protocol, end - start)
            if ipv4
            else struct.pack("!I3xB", end - start, protocol)
        )
        updated = checksum(addresses + trailer + frame[start:end])
    struct.pack_into("!H", frame, field, updated or (0xFFFF if protocol == 17 else 0))


def _ipv4(frame: bytearray, offset: int, addresses: bytes) -> None:
    start = offset + 12
    old = frame[start : start + 8]
    frame[start : start + 8] = addresses
    ihl = (frame[offset] & 15) * 4
    end = offset + struct.unpack_from("!H", frame, offset + 2)[0]
    fragment = struct.unpack_from("!H", frame, offset + 6)[0]
    segment = offset + ihl
    protocol = frame[offset + 9]
    if ihl < 20 or segment > len(frame):
        return
    frame[offset + 10 : offset + 12] = b"\0\0"
    struct.pack_into("!H", frame, offset + 10, checksum(frame[offset:segment]))

    if protocol in (6, 17) and not fragment & 0x1FFF:
        _transport_checksum(
            frame, segment, end, protocol, old, addresses, bool(fragment & 0x2000)
        )


def _ipv6(frame: bytearray, offset: int, source: Host, target: Host) -> None:
    prefix = b"\xfd" + bytes(11)
    addresses = prefix + source.ip + prefix + target.ip
    old = frame[offset + 8 : offset + 40]
    frame[offset + 8 : offset + 40] = addresses
    protocol = frame[offset + 6]
    segment = offset + 40
    end = segment + struct.unpack_from("!H", frame, offset + 4)[0]
    partial = False
    while protocol in (0, 43, 44, 60) and segment + 8 <= min(end, len(frame)):
        next_header, length = frame[segment : segment + 2]
        if protocol == 44:
            fragment = struct.unpack_from("!H", frame, segment + 2)[0]
            if fragment & 0xFFF8:
                return
            partial = bool(fragment & 1)
            segment += 8
        else:
            segment += (length + 1) * 8
        protocol = next_header
    _transport_checksum(frame, segment, end, protocol, old, addresses, partial)


def rewrite_frame(data: bytes, source: Host, target: Host) -> bytes:
    if len(data) < 14:
        raise ValueError("Cannot remap an incomplete Ethernet header")
    frame = bytearray(target.mac + source.mac + data[12:])
    ethertype, offset = ethernet_payload(data)
    if ethertype == b"\x08\x00" and len(frame) >= offset + 20:
        _ipv4(frame, offset, source.ip + target.ip)
    elif (
        ethertype == b"\x08\x06"
        and frame[offset : offset + 6] == b"\x00\x01\x08\x00\x06\x04"
        and len(frame) >= offset + 28
    ):
        frame[offset + 8 : offset + 28] = source.mac + source.ip + target.mac + target.ip
    elif ethertype == b"\x86\xdd" and len(frame) >= offset + 40:
        _ipv6(frame, offset, source, target)
    return bytes(frame)


def generate_pcap_dataset(
    path: Path,
    pools: list[list[Window]],
    pairs: list[tuple[Host, Host]],
    *,
    time_scale: float = 1,
) -> None:
    directory = path.with_suffix("")
    directory.mkdir()
    captures = []
    rates = {}
    duration = len(pools[0]) * time_scale
    for index, ((source, target), windows) in enumerate(zip(pairs, pools, strict=True)):
        capture = directory / f"{index + 1}.pcap"
        throughput = [0.0] * math.ceil(duration)
        packets = frame_bytes = 0
        first_packet = None
        with capture.open("wb") as stream:
            stream.write(PCAP_HEADER)
            for elapsed, window in enumerate(windows):
                if window.packets:
                    records = sorted(
                        capture_packets(window.capture), key=lambda packet: packet[0]
                    )
                    for stamp, length, data in records:
                        timestamp = round(
                            (elapsed * SECOND + stamp % SECOND) * time_scale
                        )
                        if first_packet is None:
                            first_packet = timestamp / SECOND
                        write_packet(
                            stream,
                            timestamp,
                            length,
                            rewrite_frame(data, source, target),
                        )
                        throughput[timestamp // SECOND] += length * 8 / 1_000_000
                        frame_bytes += length
                    packets += len(records)
        rates[source.name, target.name] = throughput
        captures.append(
            {
                "source": source.name,
                "target": target.name,
                "path": str(capture.relative_to(path.parent)),
                "packets": packets,
                "frame_bytes": frame_bytes,
                "first_packet_seconds": first_packet,
                "mbps_by_second": throughput,
            }
        )
    hosts = {host.name: host.to_dict() for pair in pairs for host in pair}
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hosts": list(hosts.values()),
                "duration_seconds": duration,
                "traffic_scale_factor": time_scale,
                "captures": captures,
            },
            indent=2,
        )
        + "\n"
    )
    write_chart(path, rates)
