"""Stream Ethernet frames from classic PCAP, PCAPNG, and gzip captures."""

import gzip
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

IPPair = tuple[bytes, bytes]

PCAP_BYTE_ORDER = {
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">",
    b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
}


def ethernet_payload(frame: bytes) -> tuple[bytes, int]:
    """Return the payload EtherType and offset after any VLAN tags."""
    offset = 14
    ethertype = frame[12:14]
    while ethertype in (b"\x81\x00", b"\x88\xa8"):
        ethertype = frame[offset + 2 : offset + 4]
        offset += 4
    return ethertype, offset


def ip_pair(frame: bytes) -> IPPair | None:
    """Read directed IPv4 or IPv6 endpoints from a complete Ethernet frame."""
    ethertype, offset = ethernet_payload(frame)
    if ethertype == b"\x08\x00" and len(frame) >= offset + 20:
        return frame[offset + 12 : offset + 16], frame[offset + 16 : offset + 20]
    if ethertype == b"\x86\xdd" and len(frame) >= offset + 40:
        return frame[offset + 8 : offset + 24], frame[offset + 24 : offset + 40]
    return None


def _pcap_packets(source: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
    """Decode classic PCAP byte order and time precision into nanosecond records."""
    header = source.read(24)
    magic = header[:4]
    endian = PCAP_BYTE_ORDER.get(magic)
    if len(header) != 24 or endian is None:
        raise ValueError("Expected a classic libpcap or PCAPNG capture")
    snaplen, link_type = struct.unpack_from(endian + "II", header, 16)
    if link_type & 0xFFFF != 1:
        raise ValueError("Expected an Ethernet capture")

    ticks_per_second = (
        1_000_000
        if magic
        in (
            b"\xd4\xc3\xb2\xa1",
            b"\xa1\xb2\xc3\xd4",
        )
        else 1_000_000_000
    )
    record = struct.Struct(endian + "IIII")
    number = 0
    while data := source.read(record.size):
        number += 1
        if len(data) != record.size:
            raise ValueError(f"Incomplete record header at packet {number}")
        seconds, fraction, captured, original = record.unpack(data)
        if captured > min(original, snaplen) or fraction >= ticks_per_second:
            raise ValueError(f"Invalid record at packet {number}")
        frame = source.read(captured)
        if len(frame) != captured:
            raise ValueError(f"Incomplete frame at packet {number}")
        timestamp_ns = seconds * 1_000_000_000 + fraction * (
            1_000_000_000 // ticks_per_second
        )
        yield timestamp_ns, original, frame


def _pcapng_packets(source: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
    """Read timestamped Ethernet PCAPNG records with per-interface time resolution."""
    from scapy.error import Scapy_Exception
    from scapy.utils import RawPcapNgReader

    try:
        with RawPcapNgReader(source) as reader:
            for frame, metadata in reader:
                if metadata.linktype != 1:
                    raise ValueError("Expected an Ethernet capture")
                if metadata.tshigh is None or metadata.tslow is None:
                    raise ValueError("PCAPNG packet has no timestamp")
                ticks = (metadata.tshigh << 32) | metadata.tslow
                yield ticks * 1_000_000_000 // metadata.tsresol, metadata.wirelen, frame
    except Scapy_Exception as error:
        raise ValueError(f"Could not read PCAPNG capture: {error}") from error


def capture_packets(capture: Path) -> Iterator[tuple[int, int, bytes]]:
    """Read Ethernet frames with their Unix nanoseconds and original lengths."""
    opener = gzip.open if capture.suffix.lower() == ".gz" else Path.open
    with opener(capture, "rb") as source:
        read_packets = (
            _pcapng_packets
            if source.peek(4)[:4] == b"\x0a\x0d\x0d\x0a"
            else _pcap_packets
        )
        yield from read_packets(source)
