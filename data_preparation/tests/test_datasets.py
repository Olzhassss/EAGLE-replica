"""Regression tests for window assignment, replay and load scaling."""

import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from random import Random

from scapy.utils import checksum

from data_preparation.analyze_trace import capture_packets
from data_preparation.load_and_split import (
    MG_PAIR,
    PCAP_HEADER,
    SECOND,
    load_windows,
    pool_windows,
    split_dc,
    write_packet,
)
from data_preparation.prepare import prepare_datasets
from data_preparation.replay import hosts_for
from data_preparation.traffic_congestions import generate_congestions


def _frame(pair: tuple[bytes, bytes], payload: bytes) -> bytes:
    segment = struct.pack("!HHHH", 101, 202, len(payload) + 8, 1) + payload
    ip = struct.pack("!BBHHHBBH", 0x45, 0, len(segment) + 20, 0, 0, 64, 17, 0)
    return bytes(12) + b"\x08\x00" + ip + b"".join(pair) + segment


def _capture(path: Path, rows: list[tuple[float, bytes]]) -> None:
    open_capture = gzip.open if path.suffix == ".gz" else Path.open
    with open_capture(path, "wb") as stream:
        stream.write(PCAP_HEADER)
        for seconds, frame in rows:
            stamp = int(seconds * SECOND)
            write_packet(stream, stamp, len(frame), frame)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


class DatasetTests(unittest.TestCase):
    def test_windows_and_pooling(self) -> None:
        isolated = _frame(MG_PAIR, b"isolated")
        other = _frame(MG_PAIR[::-1], b"other")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "example.pcap"
            _capture(capture, [(10.5, other), (11.25, isolated), (11.125, other), (13.5, other)])
            mg = load_windows(capture, root / "mg", megalodon=True)
            self.assertEqual(
                [(window.phase, window.second, window.packets) for window in mg],
                [("remaining", 11, 1), ("remaining", 12, 0),
                 ("isolated", 11, 1), ("isolated", 12, 0)],
            )
            self.assertEqual(list(capture_packets(mg[2].capture))[0][2], isolated)
            windows = load_windows(capture, root / "dc")
            train, test = split_dc(windows, Random(42))
            self.assertNotEqual(train[0].second, test[0].second)
            pools = pool_windows(mg, 2, 3, Random(42))
            picks = pools[0] + pools[1]
            self.assertEqual(len({(w.phase, w.second) for w in picks[:4]}), 4)
            self.assertEqual(pools, pool_windows(mg, 2, 3, Random(42)))

    def test_outputs_and_regional_scaling(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc, mg = root / "dc.pcap.gz", root / "mg.pcap"
            _capture(dc, [(second + 0.125, frame) for second in range(81)])
            _capture(mg, [(second + 0.125, frame) for second in range(7)])
            output = root / "3_hosts"
            prepare_datasets(dc, mg, output, 3, 20, 42, decision_seconds=5)
            train, test = output / "train", output / "test"
            self.assertFalse((train / "mg_t0.json").exists())

            dc_train = _read(train / "dc_t0.json")
            dc_test = _read(test / "dc_t0.json")
            self.assertEqual(dc_test["duration_seconds"], 100)
            self.assertEqual(dc_test["decisions"], 20)
            self.assertEqual(len(dc_test["hosts"]), 3)
            train_seconds, test_seconds = (
                {w["capture_second"] for row in profile["series"] for w in row["windows"]}
                for profile in (dc_train, dc_test)
            )
            self.assertEqual((len(train_seconds), len(test_seconds)), (63, 16))
            self.assertTrue(train_seconds.isdisjoint(test_seconds))
            self.assertEqual(len(dc_test["series"]), 6)
            self.assertTrue(all(job["rate_bps"] == job["size_bytes"] * 8 for job in dc_test["jobs"]))

            source, target, _ = hosts_for(3, None)
            entry = _read(test / "dc_pcap.json")["captures"][0]
            records = list(capture_packets(test / entry["path"]))
            self.assertEqual([stamp for stamp, _, _ in records], [
                second * SECOND + SECOND // 8 for second in range(100)
            ])
            rewritten = records[0][2]
            self.assertEqual(rewritten[:12], target.mac + source.mac)
            addresses = source.ip + target.ip
            self.assertEqual(rewritten[26:34], addresses)
            self.assertEqual(checksum(rewritten[14:34]), 0)
            self.assertEqual(
                checksum(addresses + struct.pack("!BBH", 0, 17, len(rewritten) - 34) + rewritten[34:]),
                0,
            )
            self.assertEqual(rewritten[42:], frame[42:])

            network = root / "scenario.json"
            network.write_text(json.dumps({
                "controllers": [{"id": f"c{i}"} for i in range(1, 4)],
                "switches": [{"id": f"s{i}", "controller": f"c{i}"} for i in range(1, 4)],
                "hosts": [{"id": f"h{i}"} for i in range(1, 4)],
                "links": [{"source": f"s{i}", "target": f"h{i}"} for i in range(1, 4)],
            }))
            generate_congestions(network, output, 3)
            for original, scaled in zip(
                dc_train["series"], _read(train / "dc_t0_trenv2.json")["series"], strict=True
            ):
                factor = 1.5 if original["source"] == "h1" else 1
                self.assertEqual(scaled["windows"][0]["frame_bytes"], len(frame) * factor)
            self.assertEqual(_read(test / "dc_t0_trenv5.json"), _read(test / "dc_t3.json"))
            self.assertEqual(len(list(output.glob("*/*.json"))), 17)
            for dataset in output.glob("*/*.json"):
                self.assertTrue(dataset.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
