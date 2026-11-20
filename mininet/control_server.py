"""Serve local network status and managed traffic operations."""

import json
import socket
import socketserver
import struct
import subprocess
from typing import Any

from mininet.traffic import TrialTraffic


class NetworkSession:
    """Dispatch local commands to one running network and its managed traffic."""

    def __init__(self, fingerprint: str, traffic: TrialTraffic) -> None:
        self.fingerprint = fingerprint
        self.traffic = traffic

    def command(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return network identity or reset, start, or finish the managed traffic."""
        operation = request.get("operation")
        if operation == "status":
            return {"network": self.fingerprint}
        if operation == "finish":
            return self.traffic.finish(request.get("drain_seconds", 0))
        if operation == "reset":
            self.traffic.finish()
            return {"network": self.fingerprint}
        if operation == "traffic":
            return self.traffic.start(
                request["jobs"],
                request["duration_seconds"],
                measure_flows=request.get("measure_flows", True),
            )
        raise ValueError("Unknown network operation")


class ControlRequest(socketserver.StreamRequestHandler):
    """Handle one JSON command from the session owner or root over a Unix socket."""

    server: Any

    def handle(self) -> None:
        """Check peer credentials and return a command result or serialized service error."""
        self.connection.settimeout(10)
        credentials = self.connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, 12
        )
        _, uid, _ = struct.unpack("3i", credentials)
        if uid not in (0, self.server.owner_uid):
            return
        try:
            request = json.loads(self.rfile.readline(64 * 1024 * 1024))
            response = {"result": self.server.session.command(request)}
        except (
            OSError,
            TypeError,
            KeyError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            response = {"error": str(error)}
        try:
            self.wfile.write(json.dumps(response).encode() + b"\n")
        except OSError:
            pass
