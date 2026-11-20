"""Summarize an Ethernet PCAP or PCAPNG capture by directed IP host pair."""

import argparse
import csv
import gzip
import json
import socket
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
PCAP_BYTE_ORDER = {
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">",
    b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
}
DC_CAPTURE = ROOT / "5gb-tcp-connection.pcap.gz"
MG_CAPTURE = ROOT / "megalodon.pcapng"

DEFAULT_OUTPUT_DIR = ROOT / "datasets/analyses"
PLOT_PAIR_LIMIT = 4

IPPair = tuple[bytes, bytes]
WindowCounts = dict[int, dict[IPPair | None, list[int]]]


def ethernet_payload(frame: bytes) -> tuple[bytes, int]:
    offset = 14
    ethertype = frame[12:14]
    while ethertype in (b"\x81\x00", b"\x88\xa8"):
        ethertype = frame[offset + 2 : offset + 4]
        offset += 4
    return ethertype, offset


def ip_pair(frame: bytes) -> IPPair | None:
    ethertype, offset = ethernet_payload(frame)
    if ethertype == b"\x08\x00" and len(frame) >= offset + 20:
        return frame[offset + 12 : offset + 16], frame[offset + 16 : offset + 20]
    if ethertype == b"\x86\xdd" and len(frame) >= offset + 40:
        return frame[offset + 8 : offset + 24], frame[offset + 24 : offset + 40]
    return None


def _ip_text(address: bytes) -> str:
    family = socket.AF_INET if len(address) == 4 else socket.AF_INET6
    return socket.inet_ntop(family, address)


def _stats(packets: float, frame_bytes: float, duration: float) -> dict:
    return {
        "packets": packets,
        "frame_bytes": frame_bytes,
        "average_frame_bytes_per_packet": frame_bytes / packets if packets else None,
        "average_packets_per_second": packets / duration if duration else None,
        "average_bits_per_second": 8 * frame_bytes / duration if duration else None,
    }


def _pcap_packets(source: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
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


def analyze(capture: Path, *, windows: WindowCounts | None = None) -> dict:
    """Count all records and group IP frames by source and destination address."""
    pairs: dict[IPPair, list[int]] = {}
    packets = frame_bytes = unassigned_packets = unassigned_bytes = 0
    earliest = latest = None
    for timestamp_ns, original, frame in capture_packets(capture):
        packets += 1
        if earliest is None or timestamp_ns < earliest:
            earliest = timestamp_ns
        if latest is None or timestamp_ns > latest:
            latest = timestamp_ns
        frame_bytes += original
        pair = ip_pair(frame)
        if windows is not None:
            window = windows.setdefault(timestamp_ns // 1_000_000_000, {})
            counts = window.setdefault(pair, [0, 0])
            counts[0] += 1
            counts[1] += original
        if pair is None:
            unassigned_packets += 1
            unassigned_bytes += original
        else:
            counts = pairs.setdefault(pair, [0, 0])
            counts[0] += 1
            counts[1] += original

    duration = (latest - earliest) / 1_000_000_000 if packets else 0.0
    pair_count = len(pairs)
    return {
        "capture": str(capture),
        "duration_seconds": duration,
        "total": _stats(packets, frame_bytes, duration),
        "without_ip_pair": _stats(unassigned_packets, unassigned_bytes, duration),
        "directed_pair_count": pair_count,
        "mean_per_directed_pair": _stats(
            (packets - unassigned_packets) / pair_count,
            (frame_bytes - unassigned_bytes) / pair_count,
            duration,
        )
        if pair_count
        else None,
        "directed_pairs": [
            {
                "source_ip": _ip_text(source_ip),
                "destination_ip": _ip_text(destination_ip),
                **_stats(count, byte_count, duration),
            }
            for (source_ip, destination_ip), (count, byte_count) in sorted(
                pairs.items(),
                key=lambda entry: (-entry[1][1], entry[0]),
            )
        ],
    }


def _write_window_outputs(windows: WindowCounts, report: dict, directory: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory.mkdir(parents=True, exist_ok=True)
    seconds = range(min(windows), max(windows) + 1) if windows else range(0)
    pairs = {pair for counts in windows.values() for pair in counts if pair is not None}
    series = [("total", None)]
    series.extend(("directed_pair", pair) for pair in sorted(pairs))
    if report["without_ip_pair"]["packets"]:
        series.append(("without_ip_pair", None))
    plot_pairs = {
        (pair["source_ip"], pair["destination_ip"])
        for pair in report["directed_pairs"][:PLOT_PAIR_LIMIT]
    }

    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    packet_axis, rate_axis = figure.subplots(2, 1, sharex=True)
    with (directory / "traffic_1s.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "elapsed_seconds",
                "window_start_unix_seconds",
                "scope",
                "source_ip",
                "destination_ip",
                "average_packets_per_second",
                "average_mbps",
            )
        )
        for scope, pair in series:
            source_ip = destination_ip = ""
            label = "All traffic" if scope == "total" else "Without an IP pair"
            if pair is not None:
                source_ip, destination_ip = map(_ip_text, pair)
                label = f"{source_ip} → {destination_ip}"
            plot = scope == "total" or (source_ip, destination_ip) in plot_pairs
            packet_rates = []
            bit_rates = []
            for index, second in enumerate(seconds):
                counts = windows.get(second, {})
                if scope == "total":
                    packets = sum(count[0] for count in counts.values())
                    frame_bytes = sum(count[1] for count in counts.values())
                else:
                    packets, frame_bytes = counts.get(pair, (0, 0))
                mbps = frame_bytes * 8 / 1_000_000
                writer.writerow(
                    (index, second, scope, source_ip, destination_ip, packets, mbps)
                )
                if plot:
                    packet_rates.append(packets)
                    bit_rates.append(mbps)
            linestyle = "-" if scope == "total" else "--"
            edges = range(len(seconds) + 1)
            if seconds and plot:
                packet_axis.stairs(packet_rates, edges, label=label, linestyle=linestyle)
                rate_axis.stairs(bit_rates, edges, linestyle=linestyle)

    title = "Traffic in one-second windows"
    if len(pairs) > PLOT_PAIR_LIMIT:
        title += f"\nTotal and top {PLOT_PAIR_LIMIT} directed pairs by bytes"
    figure.suptitle(title)
    packet_axis.set_ylabel("Packets/s")
    rate_axis.set_ylabel("Mbps")
    rate_axis.set_xlabel("Seconds from first window")
    for axis in (packet_axis, rate_axis):
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    if seconds:
        packet_axis.legend()
        rate_axis.set_xlim(0, len(seconds))
    figure.savefig(directory / "traffic_1s.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Rates use the full capture span for every pair. Byte counts use "
            "original Ethernet frame lengths. Zero-duration rates are null. "
            "The CSV and PNG use one-second Unix timestamp windows. "
            f"The PNG shows total traffic and up to {PLOT_PAIR_LIMIT} directed pairs by bytes."
        ),
    )
    parser.add_argument("capture", nargs="?", type=Path, default=DC_CAPTURE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="dc",
        help="directory for traffic_1s.csv and traffic_1s.png",
    )
    parser.add_argument(
        "--mg",
        action="store_true",
        help="run on Megalodon capture. Overrides `capture` and `output-dir` args.",
    )

    args = parser.parse_args()
    if args.mg:
        args.capture = MG_CAPTURE
        args.output_dir = Path("mg")
    windows: WindowCounts = {}
    try:
        report = analyze(args.capture, windows=windows)
        _write_window_outputs(windows, report, DEFAULT_OUTPUT_DIR / args.output_dir)
    except (OSError, EOFError, ImportError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
