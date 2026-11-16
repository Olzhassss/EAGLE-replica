"""Derive completion times from managed TCP transfers or paired packet captures."""

import json
from bisect import bisect_left
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class ByteRanges:
    """Track unique TCP payload sequence ranges across retransmissions."""

    def __init__(self) -> None:
        self.ranges = []
        self.size = 0

    def add(self, start: int, end: int) -> None:
        index = bisect_left(self.ranges, (start, end))
        if index > 0 and self.ranges[index - 1][1] >= start:
            index -= 1
        finish = index
        while finish < len(self.ranges) and self.ranges[finish][0] <= end:
            left, right = self.ranges[finish]
            self.size -= right - left
            start, end = min(start, left), max(end, right)
            finish += 1
        self.ranges[index:finish] = [(start, end)]
        self.size += end - start

    def add_intersection(self, start: int, end: int, expected: "ByteRanges") -> None:
        index = max(0, bisect_left(expected.ranges, (start, end)) - 1)
        while index < len(expected.ranges):
            lo, hi = expected.ranges[index]
            if lo >= end:
                break
            left, right = max(lo, start), min(hi, end)
            if left < right:
                self.add(left, right)
            index += 1


FlowKey = tuple[str, int, str, int]
CaptureCache = dict[tuple[str, str], dict[FlowKey, tuple[int, ByteRanges]]]


def collect_flow_records(
    traffic: dict[str, Any],
    capture_cache: CaptureCache,
) -> Iterator[dict[str, Any]]:
    for job in traffic.get("flows", []):
        if job["kind"] == "tcp":
            yield _managed_tcp(job)
        else:
            yield from _pcap_flows(job, capture_cache)


def _read_measurement(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def _managed_tcp(job: dict[str, Any]) -> dict[str, Any]:
    source = _read_measurement(job["sender_record"])
    target = _read_measurement(job["receiver_record"])
    started = source.get("started_at")
    completed = target.get("completed_at")
    fct = completed - started if started is not None and completed is not None else None
    first = target.get("first_byte_at")
    received = target.get("received_bytes", 0)
    return {
        "flow_id": job["id"],
        "kind": "tcp",
        "source": job["source"],
        "target": job["target"],
        "scheduled_at": job["scheduled_at"],
        "started_at": started,
        "completed_at": completed,
        "offered_bytes": job["size_bytes"],
        "sent_bytes": source.get("sent_bytes", 0),
        "received_bytes": received,
        "fct_seconds": fct,
        "first_byte_seconds": first - started
        if first is not None and started is not None
        else None,
        "sender_error": source.get("error"),
    }


def _tcp_packets(path: str | Path) -> Iterator[tuple[FlowKey, int, int, float]]:
    from scapy.layers.inet import IP, TCP
    from scapy.layers.inet6 import IPv6
    from scapy.packet import Padding
    from scapy.utils import PcapReader

    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return
    with PcapReader(str(path)) as capture:
        for packet in capture:
            if TCP not in packet:
                continue
            protocol = packet.getlayer(IP) or packet.getlayer(IPv6)
            if protocol is None:
                continue
            tcp = packet[TCP]
            padding = tcp.getlayer(Padding)
            length = len(tcp.payload) - (len(padding) if padding is not None else 0)
            key = (protocol.src, tcp.sport, protocol.dst, tcp.dport)
            sequence = (tcp.seq + int(bool(tcp.flags.S))) & 0xFFFF_FFFF
            yield key, sequence, length, float(packet.time)


def _expected_payload(path: str | Path) -> dict[FlowKey, tuple[int, ByteRanges]]:
    flows = {}
    for key, sequence, length, _ in _tcp_packets(path):
        if length:
            base, payload = flows.setdefault(key, (sequence, ByteRanges()))
            start = (sequence - base) & 0xFFFF_FFFF
            payload.add(start, start + length)
    return flows


def _pcap_flows(job: dict[str, Any], cache: CaptureCache) -> Iterator[dict[str, Any]]:
    key = (job["resolved_path"], job["sha256"])
    if key not in cache:
        cache[key] = _expected_payload(job["resolved_path"])
    expected = cache[key]
    if not expected:
        raise ValueError("No TCP payload flows in trace " + job["id"])
    sent = {}
    sent_packets = {}
    start_times = {}
    for flow, sequence, length, timestamp in _tcp_packets(job["sent_capture"]):
        if flow not in expected:
            continue
        start_times.setdefault(flow, timestamp)
        if length:
            base, payload = expected[flow]
            start = (sequence - base) & 0xFFFF_FFFF
            sent.setdefault(flow, ByteRanges()).add_intersection(
                start, start + length, payload
            )
            sent_packets.setdefault((flow, sequence, length), timestamp)
    received = {}
    first_seen = {}
    complete_at = {}
    delay_sum = {}
    delay_count = {}
    for flow, sequence, length, timestamp in _tcp_packets(job["received_capture"]):
        if length <= 0 or flow not in expected:
            continue
        sent_at = sent_packets.pop((flow, sequence, length), None)
        if sent_at is not None and sent_at <= timestamp:
            delay_sum[flow] = delay_sum.get(flow, 0.0) + timestamp - sent_at
            delay_count[flow] = delay_count.get(flow, 0) + 1
        if flow in complete_at:
            continue
        base, payload = expected[flow]
        start = (sequence - base) & 0xFFFF_FFFF
        seen = received.setdefault(flow, ByteRanges())
        before = seen.size
        seen.add_intersection(start, start + length, payload)
        if seen.size > before:
            first_seen.setdefault(flow, timestamp)
        if seen.size == payload.size:
            complete_at[flow] = timestamp
    for flow, (_, payload) in expected.items():
        started = start_times.get(flow)
        completed = complete_at.get(flow)
        first = first_seen.get(flow)
        yield {
            "flow_id": job["id"] + ":" + " ".join(map(str, flow)),
            "kind": "pcap",
            "source": job["source"],
            "target": job["target"],
            "scheduled_at": job["scheduled_at"],
            "started_at": started,
            "completed_at": completed,
            "offered_bytes": payload.size,
            "sent_bytes": sent[flow].size if flow in sent else 0,
            "received_bytes": received[flow].size if flow in received else 0,
            "fct_seconds": completed - started
            if completed is not None and started is not None
            else None,
            "first_byte_seconds": first - started
            if first is not None and started is not None
            else None,
            "matched_packets": delay_count.get(flow, 0),
            "packet_delay_sum_seconds": delay_sum.get(flow, 0),
        }
