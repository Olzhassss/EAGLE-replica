"""Run and stop the scenario's Ryu processes using the local Python 3.7 venv."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FrameType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class ControllerSupervisor:
    def __init__(self, scenario_path: Path, selected: list[str] | None) -> None:
        self.scenario_path = scenario_path.resolve()
        scenario = json.loads(self.scenario_path.read_text())
        if scenario.get("schema_version") != 1:
            raise ValueError("Expected scenario schema version 1")
        self.controllers = {row["id"]: row for row in scenario["controllers"]}
        if selected:
            if set(selected) - self.controllers.keys():
                raise ValueError("Unknown selected controller ID")
            self.controllers = {
                key: row for key, row in self.controllers.items() if key in selected
            }
        self.processes = {}

    def start(self, controller_id: str) -> None:
        process = self.processes.get(controller_id)
        if process is not None and process.poll() is None:
            return
        controller = self.controllers[controller_id]
        bind_address = controller.get("listen_host", controller["host"])
        env = dict(
            os.environ,
            EAGLE_SCENARIO=str(self.scenario_path),
            EAGLE_CONTROLLER_ID=controller_id,
        )
        self.processes[controller_id] = subprocess.Popen(
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
                "ryu_controller.controller",
            ],
            cwd=str(ROOT),
            env=env,
            start_new_session=True,
        )

    def stop(self, controller_id: str) -> None:
        process = self.processes.get(controller_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def command(self, command: dict[str, Any]) -> dict[str, dict[str, int | None]]:
        operation = command.get("operation")
        if operation == "restore":
            for controller_id in self.controllers:
                self.start(controller_id)
        elif operation in ("stop", "start"):
            controller_id = command["controller_id"]
            if controller_id not in self.controllers:
                raise ValueError("Unknown controller " + controller_id)
            handler = self.start if operation == "start" else self.stop
            handler(controller_id)
        elif operation != "status":
            raise ValueError("Unknown supervisor operation")
        return {
            name: {"pid": process.pid, "exit_code": process.poll()}
            for name, process in self.processes.items()
        }


class SupervisorRequest(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(5)
        try:
            command = json.loads(self.rfile.readline(8192))
            if not isinstance(command, dict):
                raise ValueError("Expected a JSON object")
            reply = {"result": self.server.supervisor.command(command)}
        except (KeyError, ValueError, OSError, TypeError) as error:
            reply = {"error": str(error)}
        self.wfile.write(json.dumps(reply).encode() + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios/v0.json")
    parser.add_argument(
        "--controller",
        action="append",
        help="Launch only this controller ID on this host",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 7) or sys.prefix == sys.base_prefix:
        parser.error("Launch from the controller's Python 3.7 virtual environment")
    supervisor = ControllerSupervisor(args.scenario, args.controller)

    def stop_on_signal(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        with tempfile.TemporaryDirectory(prefix="eagle-ryu-") as directory:
            address = str(Path(directory) / "control.sock")
            with socketserver.UnixStreamServer(address, SupervisorRequest) as server:
                server.supervisor = supervisor
                for controller_id in supervisor.controllers:
                    supervisor.start(controller_id)
                print("Ryu supervisor socket: " + address, flush=True)
                server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        for controller_id in supervisor.controllers:
            supervisor.stop(controller_id)


if __name__ == "__main__":
    main()
