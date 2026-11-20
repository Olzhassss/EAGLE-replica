"""Send commands to the local Mininet session and Ryu supervisor."""

import json
import socket
from pathlib import Path
from typing import Any

from EAGLE.src.paths import ROOT


class SocketCommander:
    """Discover service sockets and exchange newline-delimited JSON commands."""

    def __init__(
        self,
        mininet_socket: Path | str | None = None,
        ryu_socket: Path | str | None = None,
        *,
        discover: bool = True,
    ) -> None:
        """Resolve explicit socket overrides or read addresses written by the launchers."""
        addresses = []
        for override, address_file in (
            (mininet_socket, ROOT / "mininet/.socket"),
            (ryu_socket, ROOT / ".socket"),
        ):
            address = override
            if address is None and discover and address_file.exists():
                address = address_file.read_text().strip()
            addresses.append(Path(address) if address else None)
        self.mininet_socket, self.ryu_socket = addresses

    def _unix_command(
        self,
        path: Path | None,
        operation: str,
        *,
        timeout: float = 300,
        **arguments: Any,
    ) -> Any:
        """Send one command, read its response, and propagate a service error."""
        if path is None:
            raise ValueError(f"A control socket is required for {operation}")
        payload = json.dumps(dict(arguments, operation=operation)).encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(path))
            connection.sendall(payload)
            with connection.makefile("rb") as stream:
                data = json.loads(stream.readline())
        if "error" in data:
            raise RuntimeError(f"{operation} via {path}: {data['error']}")
        return data["result"]

    def status(self) -> dict[str, Any]:
        """Return the fingerprint of the network running in Mininet."""
        return self._unix_command(self.mininet_socket, "status")

    def reset(self, fingerprint: str) -> None:
        """Check network identity, restore controllers, and stop managed traffic."""
        if self.status()["network"] != fingerprint:
            raise ValueError("Mininet network differs from the selected network file")
        if self.ryu_socket:
            self._unix_command(self.ryu_socket, "restore")
        self._unix_command(self.mininet_socket, "reset")

    def start_traffic(
        self,
        jobs: list[dict],
        duration: float,
        *,
        measure_flows: bool = True,
    ) -> dict[str, Any]:
        """Start a prepared job schedule and optionally collect flow measurements."""
        return self._unix_command(
            self.mininet_socket,
            "traffic",
            jobs=jobs,
            duration_seconds=duration,
            measure_flows=measure_flows,
        )

    def finish_traffic(self, drain_seconds: float = 0) -> dict[str, Any]:
        """Drain active flows, stop traffic processes, and return their artifacts."""
        return self._unix_command(
            self.mininet_socket,
            "finish",
            drain_seconds=drain_seconds,
            timeout=drain_seconds + 300,
        )
