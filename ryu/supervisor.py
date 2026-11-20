"""Run and stop the network's Ryu processes using the local Python 3.7 venv."""

import argparse
import json
import os
import signal
import socketserver
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from network_schema import validate_support_layout

ROOT = Path(__file__).resolve().parents[1]
ADDRESS_FILE = ROOT / ".socket"


class ControllerSupervisor:
    """Manage the local Ryu processes for the selected network controllers."""

    def __init__(self, network_path: Path, selected: Optional[List[str]]) -> None:
        self.network_path = network_path.resolve()
        network = json.loads(self.network_path.read_text())
        validate_support_layout(network)
        self.controllers = {row["id"]: row for row in network["controllers"]}
        if selected:
            if set(selected) - self.controllers.keys():
                raise ValueError("Unknown selected controller ID")
            self.controllers = {
                key: row for key, row in self.controllers.items() if key in selected
            }
        self.processes = {}

    def start(self, controller_id: str) -> None:
        """Start a missing controller process with the network identity in its environment."""
        process = self.processes.get(controller_id)
        if process is not None and process.poll() is None:
            return
        controller = self.controllers[controller_id]
        bind_address = controller.get("listen_host", controller["host"])
        env = dict(
            os.environ,
            EAGLE_NETWORK=str(self.network_path),
            EAGLE_CONTROLLER_ID=controller_id,
        )
        self.processes[controller_id] = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                "from ryu.cmd.manager import main; main()",
                "--ofp-listen-host",
                bind_address,
                "--ofp-tcp-listen-port",
                str(controller["openflow_port"]),
                "--wsapi-host",
                bind_address,
                "--wsapi-port",
                str(controller["http_port"]),
                "ryu.eagle_controller",
            ],
            cwd=str(ROOT),
            env=env,
            start_new_session=True,
        )

    def stop(self, controller_id: str) -> None:
        """Terminate and reap a controller, killing it if graceful shutdown times out."""
        process = self.processes.get(controller_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def command(self, command: Dict[str, Any]) -> Dict[str, Dict[str, Optional[int]]]:
        """Restore missing controllers or report each managed process state."""
        operation = command.get("operation")
        if operation == "restore":
            for controller_id in self.controllers:
                self.start(controller_id)
        elif operation != "status":
            raise ValueError("Unknown supervisor operation")
        return {
            name: {"pid": process.pid, "exit_code": process.poll()}
            for name, process in self.processes.items()
        }


class SupervisorRequest(socketserver.StreamRequestHandler):
    """Dispatch one local supervisor command and return a JSON result or error."""

    def handle(self) -> None:
        self.connection.settimeout(5)
        try:
            command = json.loads(self.rfile.readline(8192))
            reply = {"result": self.server.supervisor.command(command)}
        except (KeyError, ValueError, OSError, TypeError) as error:
            reply = {"error": str(error)}
        self.wfile.write(json.dumps(reply).encode() + b"\n")


def main() -> None:
    """Run the supervisor socket and stop every managed controller on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=ROOT / "networks/v0.json")
    parser.add_argument(
        "--controller",
        action="append",
        help="Launch only this controller ID on this host",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 7) or sys.prefix == sys.base_prefix:
        parser.error("Launch from the controller's Python 3.7 virtual environment")
    supervisor = ControllerSupervisor(args.network, args.controller)

    def stop_on_signal(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    address = None
    try:
        with tempfile.TemporaryDirectory(prefix="eagle-ryu-") as directory:
            address = str(Path(directory) / "control.sock")
            with socketserver.UnixStreamServer(address, SupervisorRequest) as server:
                server.supervisor = supervisor
                for controller_id in supervisor.controllers:
                    supervisor.start(controller_id)
                ADDRESS_FILE.write_text(address)
                server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        for controller_id in supervisor.controllers:
            supervisor.stop(controller_id)
        if ADDRESS_FILE.exists() and ADDRESS_FILE.read_text() == address:
            ADDRESS_FILE.unlink()


if __name__ == "__main__":
    main()
