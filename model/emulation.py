"""Control the local emulator and controller supervisor."""

import json
import socket
from pathlib import Path
from typing import Any

from .scenario import ROOT


def _socket_address(override: Path | str | None, address_file: Path | None) -> Path | None:
    if override is not None:
        return Path(override)
    if address_file is None:
        return None
    try:
        address = address_file.read_text().strip()
    except FileNotFoundError:
        return None
    if not address:
        raise ValueError(f"No control socket address in {address_file}")
    return Path(address)


def unix_command(path: Path | str | None, operation: str, *, timeout: float = 300, **arguments: Any) -> Any:
    if path is None:
        raise ValueError("A control socket is required for " + operation)
    payload = json.dumps(dict(arguments, operation=operation)).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(payload)
        with connection.makefile("rb") as stream:
            data = json.loads(stream.readline())
    if "error" in data:
        msg = "{} via {}: {}".format(operation, path, data["error"])
        raise RuntimeError(msg)
    return data["result"]


class Emulation:
    def __init__(
        self,
        mininet_socket: Path | str | None = None,
        ryu_socket: Path | str | None = None,
        *,
        discover: bool = True,
    ) -> None:
        self.mininet_socket = _socket_address(
            mininet_socket, ROOT / "mininet_nets/.socket" if discover else None
        )
        self.ryu_socket = _socket_address(
            ryu_socket, ROOT / ".socket" if discover else None
        )

    def reset(self, fingerprint: str) -> None:
        status = unix_command(self.mininet_socket, "status")
        if status["scenario"] != fingerprint:
            raise ValueError("Mininet network differs from the selected network file")
        if self.ryu_socket:
            unix_command(self.ryu_socket, "restore")
        unix_command(self.mininet_socket, "reset")

    def start_traffic(
        self, jobs: list[dict], duration: float, *, measure_flows: bool = True,
    ) -> dict[str, Any]:
        return unix_command(
            self.mininet_socket, "traffic", jobs=jobs,
            duration_seconds=duration, measure_flows=measure_flows,
        )

    def finish_traffic(self, drain_seconds: float = 0) -> dict[str, Any]:
        return unix_command(self.mininet_socket, "finish", drain_seconds=drain_seconds, timeout=drain_seconds + 300)
