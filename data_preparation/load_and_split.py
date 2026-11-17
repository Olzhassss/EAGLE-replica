"""Collect whole-second capture windows and assign them to directed host pairs."""

import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from socket import inet_aton
from typing import BinaryIO

from .analyze_trace import capture_packets, ip_pair

SECOND = 1_000_000_000
MG_PAIR = (inet_aton("192.168.1.1"), inet_aton("172.28.190.238"))
PCAP_HEADER = struct.pack("<IHHIIII", 0xA1B23C4D, 2, 4, 0, 0, 262144, 1)
PCAP_RECORD = struct.Struct("<IIII")
OPEN_WINDOWS = 16


@dataclass
class Window:
    second: int
    phase: str
    capture: Path
    packets: int = 0
    frame_bytes: int = 0


def write_packet(stream: BinaryIO, timestamp_ns: int, length: int, data: bytes) -> None:
    second, fraction = divmod(timestamp_ns, SECOND)
    stream.write(PCAP_RECORD.pack(second, fraction, len(data), length) + data)


def load_windows(capture: Path, spool: Path, *, megalodon: bool = False) -> list[Window]:
    """Spool original frames to disk, keeping only window counts in memory."""
    spool.mkdir()
    windows: dict[tuple[str, int], Window] = {}
    streams: OrderedDict[tuple[str, int], BinaryIO] = OrderedDict()
    earliest = latest = None
    try:
        for stamp, length, data in capture_packets(capture):
            earliest = stamp if earliest is None else min(earliest, stamp)
            latest = stamp if latest is None else max(latest, stamp)
            phase = "dc"
            if megalodon:
                phase = "isolated" if ip_pair(data) == MG_PAIR else "remaining"
            second = stamp // SECOND
            key = phase, second
            if key not in windows:
                windows[key] = Window(second, phase, spool / f"{phase}-{second}.pcap")
            window = windows[key]
            stream = streams.pop(key, None)
            if stream is None:
                if len(streams) >= OPEN_WINDOWS:
                    streams.popitem(last=False)[1].close()
                stream = window.capture.open("ab")
                if not window.packets:
                    stream.write(PCAP_HEADER)
            streams[key] = stream
            write_packet(stream, stamp, length, data)
            window.packets += 1
            window.frame_bytes += length
    finally:
        for stream in streams.values():
            stream.close()

    if earliest is None or latest is None:
        raise ValueError(f"{capture} has no packets")
    start = (earliest + SECOND - 1) // SECOND
    stop = latest // SECOND
    if start >= stop:
        raise ValueError(f"{capture} has no complete one-second windows")
    phases = ("remaining", "isolated") if megalodon else ("dc",)
    return [
        windows.get((phase, second)) or Window(second, phase, spool / f"{phase}-{second}.pcap")
        for phase in phases
        for second in range(start, stop)
    ]


def split_dc(windows: list[Window], rng: Random) -> tuple[list[Window], list[Window]]:
    if len(windows) < 2:
        raise ValueError("Digital Corpora needs at least two windows for train and test")
    shuffled = rng.sample(windows, len(windows))
    boundary = 4 * len(shuffled) // 5
    return shuffled[:boundary], shuffled[boundary:]


def pool_windows(
    windows: list[Window], pairs: int, steps: int, rng: Random
) -> list[list[Window]]:
    """Draw a new shuffled pool only after exhausting the current one."""
    if not windows:
        raise ValueError("Cannot pool windows from an empty split")
    needed = pairs * steps
    selected = []
    while len(selected) < needed:
        selected.extend(rng.sample(windows, min(len(windows), needed - len(selected))))
    return [selected[start : start + steps] for start in range(0, needed, steps)]
