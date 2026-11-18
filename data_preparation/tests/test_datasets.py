"""Regression tests for window assignment, replay and load scaling."""

import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from random import Random
from unittest.mock import patch

from scapy.utils import checksum
from config import Config

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
from data_preparation.prepare import main, prepare_datasets
from data_preparation.replay import generate_pcap_dataset, hosts_for
from data_preparation.traffic_loads import generate_synthetic_dataset, scale_synthetic
from data_preparation.traffic_congestions import generate_congestions
from model.datasets import load_dataset
from model.scenario import Scenario


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
    def test_cli_forwards_output_options(self) -> None:
        with patch("sys.argv", ["prepare", "default", "--hosts", "3", "--skip-pcaps", "--skip-train"]), \
             patch("data_preparation.prepare.prepare_datasets") as prepare:
            main()
        self.assertEqual(prepare.call_args.kwargs, {"skip_pcaps": True, "skip_train": True})
        with patch("sys.argv", ["prepare", "congestion", "--hosts", "3", "--network", "network.json", "--skip-train"]), \
             patch("data_preparation.prepare.generate_congestions") as congestion:
            main()
        self.assertEqual(congestion.call_args.kwargs, {"skip_train": True})

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

    @patch.object(Config, "TRAFFIC_SCALE_FACTOR", 1)
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

    def test_time_stretch_preserves_packets_bytes_and_synthetic_volume(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "source.pcap"
            _capture(capture, [(time, frame) for time in (0.1, 1.25, 1.75, 2.25, 2.75, 3.1)])
            windows = load_windows(capture, root / "windows")
            pools = [windows]
            pairs = [tuple(hosts_for(2))]
            with patch("data_preparation.replay.write_chart"), \
                 patch("data_preparation.traffic_loads.write_chart"):
                generate_pcap_dataset(root / "original.json", pools, pairs)
                generate_pcap_dataset(root / "stretched.json", pools, pairs, time_scale=1.5)
                generate_synthetic_dataset(root / "synthetic.json", pools, pairs, time_scale=1.5)
                generate_synthetic_dataset(root / "slow.json", pools, pairs, time_scale=90)
                scale_synthetic(root / "synthetic.json", root / "hotspot.json", 2)
            original = list(capture_packets(root / "original/1.pcap"))
            stretched = list(capture_packets(root / "stretched/1.pcap"))
            self.assertEqual([stamp for stamp, _, _ in stretched], [round(stamp * 1.5) for stamp, _, _ in original])
            self.assertEqual([(length, data) for _, length, data in original],
                             [(length, data) for _, length, data in stretched])
            manifest = _read(root / "stretched.json")
            self.assertEqual(manifest["duration_seconds"], 3)
            self.assertEqual(manifest["captures"][0]["first_packet_seconds"], 0.375)
            self.assertEqual(manifest["captures"][0]["packets"], 4)
            bits_per_frame_mbps = len(frame) * 8 / 1_000_000
            self.assertEqual(manifest["captures"][0]["mbps_by_second"],
                             [bits_per_frame_mbps, 2 * bits_per_frame_mbps, bits_per_frame_mbps])
            synthetic = _read(root / "synthetic.json")
            self.assertEqual(synthetic["duration_seconds"], 3)
            self.assertEqual(synthetic["window_seconds"], 1.5)
            self.assertEqual(sum(job["size_bytes"] for job in synthetic["jobs"]), 4 * len(frame))
            self.assertEqual([job["start_seconds"] for job in synthetic["jobs"]], [0, 1.5])
            for job in synthetic["jobs"]:
                self.assertAlmostEqual(job["size_bytes"] * 8 / job["rate_bps"], 1.5)
            self.assertAlmostEqual(synthetic["series"][0]["windows"][0]["packets_per_second"], 2 / 1.5)
            hotspot = _read(root / "hotspot.json")
            self.assertEqual(hotspot["duration_seconds"], synthetic["duration_seconds"])
            self.assertEqual(hotspot["window_seconds"], 1.5)
            self.assertEqual(hotspot["jobs"][0]["rate_bps"], synthetic["jobs"][0]["rate_bps"] * 2)
            self.assertGreater(_read(root / "slow.json")["jobs"][0]["timeout_seconds"], 90)

    def test_statistics_only_loading_matches_spooled_windows(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "source.pcap"
            _capture(capture, [(second + 0.125, frame) for second in range(6)])
            with patch("data_preparation.load_and_split.write_packet", side_effect=AssertionError("spooling")):
                statistics = load_windows(capture, root / "unused", megalodon=True, store_packets=False)
            spooled = load_windows(capture, root / "spooled", megalodon=True)
            self.assertFalse((root / "unused").exists())
            fields = lambda rows: [(row.second, row.phase, row.packets, row.frame_bytes) for row in rows]
            self.assertEqual(fields(statistics), fields(spooled))

    @patch.object(Config, "TRAFFIC_SCALE_FACTOR", None)
    def test_host_count_scaling_and_synthetic_only_preparation(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc, mg = root / "dc.pcap", root / "mg.pcap"
            _capture(dc, [(second + 0.125, frame) for second in range(21)])
            _capture(mg, [(second + 0.125, frame) for second in range(7)])
            output = root / "datasets/3_hosts"
            with patch("data_preparation.load_and_split.write_packet", side_effect=AssertionError("spooling")), \
                 patch("data_preparation.prepare.generate_pcap_dataset") as pcaps, \
                 patch("data_preparation.traffic_loads.write_chart"):
                prepare_datasets(dc, mg, output, 3, 20, 42, skip_pcaps=True)
            pcaps.assert_not_called()
            data = _read(output / "train/dc_t0.json")
            self.assertEqual(data["traffic_scale_factor"], 3)
            self.assertEqual(data["duration_seconds"], 102)
            self.assertEqual(data["decisions"], 20)
            network = Scenario(root / "network.json", {"hosts": data["hosts"]}, "network-hash")
            runtime = load_dataset(network, "dc_t0", "train", root=root)
            self.assertEqual(runtime.decisions({"warmup_seconds": 0, "decision_seconds": 5}), 20)
            self.assertEqual(runtime.duration, 102)
            self.assertEqual(len(data["series"][0]["windows"]), 34)
            self.assertEqual(len(list(output.glob("*/*.json"))), 6)
            self.assertEqual(list(output.rglob("*.pcap")), [])
            self.assertEqual(list(output.glob("*/*_pcap.json")), [])

    @patch.object(Config, "TRAFFIC_SCALE_FACTOR", 2)
    def test_skip_train_preserves_test_data_and_supports_test_only_congestion(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc, mg = root / "dc.pcap", root / "mg.pcap"
            _capture(dc, [(second + 0.125, frame) for second in range(21)])
            _capture(mg, [(second + 0.125, frame) for second in range(7)])
            with patch("data_preparation.replay.write_chart"), \
                 patch("data_preparation.traffic_loads.write_chart"):
                prepare_datasets(dc, mg, root / "full", 3, 20, 42)
                prepare_datasets(dc, mg, root / "test-only", 3, 20, 42, skip_pcaps=True, skip_train=True)
                for path in (root / "test-only/test").glob("*.json"):
                    self.assertEqual(_read(path), _read(root / "full/test" / path.name))
                network = root / "network.json"
                network.write_text(json.dumps({
                    "controllers": [{"id": f"c{i}"} for i in range(3)],
                    "switches": [{"id": f"s{i}", "controller": f"c{i}"} for i in range(3)],
                    "hosts": [{"id": f"h{i + 1}"} for i in range(3)],
                    "links": [{"source": f"s{i}", "target": f"h{i + 1}"} for i in range(3)],
                }))
                generate_congestions(network, root / "test-only", 3, skip_train=True)
            self.assertFalse((root / "test-only/train").exists())
            self.assertEqual(len(list((root / "test-only/test").glob("*.json"))), 9)
            data = _read(root / "test-only/test/dc_t0.json")
            self.assertEqual(data["traffic_scale_factor"], 2)
            self.assertEqual(data["duration_seconds"], 100)

    @patch.object(Config, "TRAFFIC_SCALE_FACTOR", 60)
    def test_scaled_available_windows_are_used_without_minimum_padding(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc, mg = root / "dc.pcap", root / "mg.pcap"
            _capture(dc, [(second + 0.125, frame) for second in range(81)])
            _capture(mg, [(second + 0.125, frame) for second in range(7)])
            output = root / "datasets"
            with patch("data_preparation.traffic_loads.write_chart"):
                prepare_datasets(dc, mg, output, 3, 20, 42, skip_pcaps=True)
            for split, source_seconds, duration, decisions in (
                ("train", 10, 600, 120), ("test", 2, 120, 24),
            ):
                data = _read(output / split / "dc_t0.json")
                self.assertEqual(data["duration_seconds"], duration)
                self.assertEqual(data["decisions"], decisions)
                self.assertTrue(all(len(row["windows"]) == source_seconds for row in data["series"]))
                selected = [window["capture_second"] for row in data["series"] for window in row["windows"]]
                self.assertEqual(len(selected), len(set(selected)))

    def test_minimum_after_scaling_covers_fractional_and_rounding_boundaries(self) -> None:
        frame = _frame(MG_PAIR, b"payload")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc, mg = root / "dc.pcap", root / "mg.pcap"
            _capture(dc, [(second + 0.125, frame) for second in range(21)])
            _capture(mg, [(second + 0.125, frame) for second in range(7)])
            cases = ((1.5, 5, 20, 67, 100.5), (1, 1.1, 30, 34, 34))
            for scale, interval, minimum, source_seconds, duration in cases:
                with self.subTest(scale=scale, interval=interval), \
                     patch.object(Config, "TRAFFIC_SCALE_FACTOR", scale), \
                     patch("data_preparation.traffic_loads.write_chart"), \
                     patch("data_preparation.replay.write_chart"):
                    output = root / f"scaled-{scale}"
                    prepare_datasets(dc, mg, output, 3, minimum, 42, decision_seconds=interval)
                    for split in ("train", "test"):
                        synthetic = _read(output / split / "dc_t0.json")
                        pcap = _read(output / split / "dc_pcap.json")
                        self.assertEqual(synthetic["duration_seconds"], duration)
                        self.assertEqual(pcap["duration_seconds"], duration)
                        self.assertGreaterEqual(synthetic["decisions"], minimum)
                        self.assertEqual(pcap["decisions"], synthetic["decisions"])
                        self.assertEqual(len(synthetic["series"][0]["windows"]), source_seconds)

    def test_invalid_time_scale_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "datasets"
            for scale in (0, 0.5, -1, True, float("inf"), float("nan"), "2"):
                with self.subTest(scale=scale), patch.object(Config, "TRAFFIC_SCALE_FACTOR", scale):
                    with self.assertRaisesRegex(ValueError, "TRAFFIC_SCALE_FACTOR"):
                        prepare_datasets(Path("dc"), Path("mg"), output, 3, 20, 42)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
