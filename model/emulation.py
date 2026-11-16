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


def unix_command(path: Path | str | None, operation: str, **arguments: Any) -> Any:
    if path is None:
        raise ValueError("A control socket is required for " + operation)
    payload = json.dumps(dict(arguments, operation=operation)).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(300 if operation == "traffic" else 15)
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

    def reset(self, fingerprint: str, mode: str, traffic: str) -> None:
        if self.mininet_socket:
            status = unix_command(self.mininet_socket, "status")
            if status["scenario"] != fingerprint or (
                traffic != "external" and traffic not in status["traffic_profiles"]
            ):
                raise ValueError(
                    "Mininet scenario or traffic profile does not match the trial",
                )
        elif mode != "forwarding" or traffic not in ("idle", "external"):
            raise ValueError(
                "Start Mininet to create mininet_nets/.socket or pass --mininet-socket",
            )
        if self.ryu_socket:
            unix_command(self.ryu_socket, "restore")
        if self.mininet_socket:
            unix_command(self.mininet_socket, "reset", resource_mode=mode)

    def start_traffic(self, profile: str) -> dict[str, Any]:
        return (
            unix_command(self.mininet_socket, "traffic", profile=profile)
            if self.mininet_socket and profile != "external"
            else {}
        )

    def finish_traffic(self) -> dict[str, Any]:
        return (
            unix_command(self.mininet_socket, "finish") if self.mininet_socket else {}
        )

    def apply_resources(self, actions: dict[str, int], mode: str) -> dict[str, Any]:
        if mode != "forwarding":
            return unix_command(self.mininet_socket, "resources", actions=actions)
        return {"mode": mode}

    def fault(self, event: dict[str, Any]) -> dict[str, Any]:
        if event["kind"] == "link":
            return unix_command(
                self.mininet_socket,
                "link",
                source=event["source"],
                target=event["target"],
                up=event["up"],
            )
        operation = "start" if event["up"] else "stop"
        return unix_command(
            self.ryu_socket,
            operation,
            controller_id=event["controller_id"],
        )
