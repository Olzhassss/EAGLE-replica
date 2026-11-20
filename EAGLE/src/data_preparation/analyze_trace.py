"""Summarize an Ethernet PCAP or PCAPNG capture by directed IP host pair."""

import argparse
import csv
import json
import socket
from pathlib import Path

from EAGLE.src.data_preparation.captures import IPPair, capture_packets, ip_pair
from EAGLE.src.paths import DC_CAPTURE, MG_CAPTURE, ROOT

DEFAULT_OUTPUT_DIR = ROOT / "datasets/analyses"
PLOT_PAIR_LIMIT = 4

WindowCounts = dict[int, dict[IPPair | None, list[int]]]


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
    """Write per-second traffic totals and directed-pair series as CSV and PNG."""
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
    """Analyze a capture and save its one-second traffic history and JSON report."""
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
    report = analyze(args.capture, windows=windows)
    _write_window_outputs(windows, report, DEFAULT_OUTPUT_DIR / args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
