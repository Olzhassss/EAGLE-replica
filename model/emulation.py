"""Control an explicitly selected local emulator and controller supervisor."""

import json
from pathlib import Path
import socket


def unix_command(path, operation, **arguments):
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
        raise RuntimeError("{} via {}: {}".format(operation, path, data["error"]))
    return data["result"]


class Emulation:
    def __init__(self, mininet_socket=None, ryu_socket=None):
        self.mininet_socket = Path(mininet_socket) if mininet_socket is not None else None
        self.ryu_socket = Path(ryu_socket) if ryu_socket is not None else None

    def reset(self, fingerprint, mode, traffic):
        if self.ryu_socket:
            unix_command(self.ryu_socket, "restore")
        if not self.mininet_socket:
            if mode != "forwarding" or traffic not in ("idle", "external"):
                raise ValueError("Managed traffic and hardware changes require --mininet-socket")
            return
        status = unix_command(self.mininet_socket, "status")
        if status["scenario"] != fingerprint or traffic not in status["traffic_profiles"]:
            raise ValueError("Mininet scenario or traffic profile does not match the trial")
        unix_command(self.mininet_socket, "reset", resource_mode=mode)

    def start_traffic(self, profile):
        return unix_command(self.mininet_socket, "traffic", profile=profile) if self.mininet_socket else {}

    def finish_traffic(self):
        return unix_command(self.mininet_socket, "finish") if self.mininet_socket else {}

    def apply_resources(self, actions, mode):
        if mode != "forwarding":
            return unix_command(self.mininet_socket, "resources", actions=actions)
        return {"mode": mode}

    def fault(self, event):
        if event["kind"] == "link":
            return unix_command(self.mininet_socket, "link", source=event["source"],
                                target=event["target"], up=event["up"])
        operation = "start" if event["up"] else "stop"
        return unix_command(self.ryu_socket, operation, controller_id=event["controller_id"])
