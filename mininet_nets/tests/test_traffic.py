"""Exercise traffic contracts with host processes and clocks mocked."""

import hashlib
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mininet_nets.launch_topology import EmulationSession
from mininet_nets.traffic import TrialTraffic
from mininet_nets.traffic_flow import receive, send


class TrafficTests(unittest.TestCase):
    def test_invalid_next_dataset_can_be_retried_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traffic = TrialTraffic(MagicMock(), root, os.getuid())
            traffic.directory = root / "previous"
            traffic.directory.mkdir()
            with (
                patch.object(
                    traffic, "_prepare", side_effect=[ValueError("invalid"), []]
                ),
                patch("mininet_nets.traffic.time.sleep"),
            ):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    traffic.start([], 100)
                self.assertIsNone(traffic.directory)
                traffic.start([], 100)
                self.assertTrue(traffic.directory.exists())

    def test_async_receiver_records_payload_completion(self) -> None:
        reader = MagicMock()
        reader.readline = AsyncMock(
            return_value=b'{"id":"flow","size_bytes":6,"timeout_seconds":5}\n'
        )
        reader.read = AsyncMock(side_effect=[b"abc", b"def"])
        writer = MagicMock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asyncio.run(receive(reader, writer, root))
            result = json.loads((root / "flow.received.json").read_text())
        self.assertEqual(result["received_bytes"], 6)
        self.assertIsNotNone(result["completed_at"])

    def test_async_sender_flushes_payload_and_writes_measurements(self) -> None:
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        job = {
            "id": "flow",
            "address": "10.0.0.2",
            "scheduled_at": 0,
            "size_bytes": 6,
            "rate_bps": 48,
            "timeout_seconds": 5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "mininet_nets.traffic_flow.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(None, writer),
            ):
                asyncio.run(send(job, root))
            result = json.loads((root / "flow.sent.json").read_text())
        self.assertEqual(result["sent_bytes"], 6)
        self.assertIsNone(result["error"])
        writer.wait_closed.assert_awaited_once()

    def test_dynamic_jobs_and_drain_reach_the_traffic_runner(self) -> None:
        traffic = MagicMock()
        session = EmulationSession("network", traffic)
        jobs = [{"id": "example"}]
        session.command({"operation": "traffic", "jobs": jobs, "duration_seconds": 100})
        traffic.start.assert_called_once_with(jobs, 100, measure_flows=True)
        session.command({"operation": "finish", "drain_seconds": 5})
        traffic.finish.assert_called_once_with(5)

    def test_pcap_replay_uses_original_timing_and_drains_before_stop(self) -> None:
        host = MagicMock()
        host.name = "h1"
        target = MagicMock()
        target.name = "h2"
        network = MagicMock()
        network.hosts = [host, target]
        network.__getitem__.side_effect = {"h1": host, "h2": target}.__getitem__
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "prepared.pcap"
            capture.write_bytes(b"fixture")
            traffic = TrialTraffic(network, root, os.getuid())
            job = {
                "id": "replay",
                "kind": "pcap",
                "source": "h1",
                "target": "h2",
                "start_seconds": 0,
                "path": str(capture),
                "sha256": hashlib.sha256(b"fixture").hexdigest(),
            }
            with (
                patch("mininet_nets.traffic.shutil.which", return_value="program"),
                patch.object(traffic, "_spawn") as spawn,
                patch("mininet_nets.traffic.time.sleep") as sleep,
            ):
                traffic.start([job], 100)
                command = spawn.call_args_list[-1].args[1]
                self.assertIn("--multiplier=1", command)
                self.assertFalse(
                    any(str(argument).startswith("--mbps") for argument in command)
                )
                process = MagicMock()
                process.poll.return_value = None
                traffic.processes = [(process, root / "log", "sender")]
                traffic.finish(5)
                sleep.assert_called_with(5)
                process.send_signal.assert_called_once()
                spawn.reset_mock()
                traffic.start([job], 100, measure_flows=False)
                self.assertEqual(spawn.call_count, 1)
                self.assertEqual(spawn.call_args.args[3], "sender")
                self.assertEqual(traffic.finish()["flows"], [])


if __name__ == "__main__":
    unittest.main()
