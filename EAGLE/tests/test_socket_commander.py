"""Exercise the socket protocol against a real local command server."""

import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from EAGLE.src.communication.socket_commander import SocketCommander


class CommandHandler(socketserver.StreamRequestHandler):
    def handle(self):
        request = json.loads(self.rfile.readline())
        self.server.requests.append(request)
        reply = (
            {"error": "traffic failed"}
            if request["operation"] == "traffic"
            else {"result": {"network": "fingerprint"}}
        )
        self.wfile.write(json.dumps(reply).encode() + b"\n")


class SocketCommanderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "control.sock"
        self.server = socketserver.UnixStreamServer(str(path), CommandHandler)
        self.server.requests = []
        self.addCleanup(self.server.server_close)
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(self.server.shutdown)
        self.commander = SocketCommander(path, path, discover=False)

    def test_status_and_reset_use_the_network_protocol(self):
        self.assertEqual(self.commander.status(), {"network": "fingerprint"})
        self.commander.reset("fingerprint")
        self.assertEqual(
            [row["operation"] for row in self.server.requests],
            ["status", "status", "restore", "reset"],
        )

    def test_mismatched_network_does_not_restore_or_reset(self):
        with self.assertRaisesRegex(ValueError, "network differs"):
            self.commander.reset("another-network")
        self.assertEqual(self.server.requests, [{"operation": "status"}])

    def test_service_error_reaches_the_caller(self):
        with self.assertRaisesRegex(RuntimeError, "traffic failed"):
            self.commander.start_traffic([], 10)
        self.assertEqual(
            self.server.requests[0],
            {
                "operation": "traffic",
                "jobs": [],
                "duration_seconds": 10,
                "measure_flows": True,
            },
        )
