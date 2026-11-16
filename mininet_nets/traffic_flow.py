"""Run one scheduled TCP transfer or an explicitly configured replay command."""

import argparse
import json
import os
import signal
import socket
import time
from types import FrameType


def send_tcp(
    host: str,
    port: int,
    size_bytes: int,
    rate_bps: int,
    starts_at: float,
    timeout: float,
    result_path: str,
) -> None:
    time.sleep(max(0.0, starts_at - time.time()))
    started = time.time()
    sent = 0
    error = None
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            deadline = time.monotonic() + timeout
            data = bytes(64 * 1024)
            while sent < size_bytes:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("TCP transfer timed out")
                connection.settimeout(budget)
                sent += connection.send(data[: min(len(data), size_bytes - sent)])
                wait = started + sent * 8 / rate_bps - time.time()
                if wait > 0 and sent < size_bytes:
                    time.sleep(min(wait, max(0, deadline - time.monotonic())))
    except OSError as exc:
        error = str(exc)
    with open(result_path, "w") as output:
        json.dump({"started_at": started, "sent_bytes": sent, "error": error}, output)


def receive_tcp(
    port: int, expected_bytes: int, timeout: float, result_path: str
) -> None:
    received = 0
    completed = None
    first_byte = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(1)
        server.settimeout(timeout)
        try:
            connection, _ = server.accept()
            with connection:
                deadline = time.monotonic() + timeout
                while received < expected_bytes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    connection.settimeout(remaining)
                    data = connection.recv(65536)
                    if not data:
                        break
                    if first_byte is None:
                        first_byte = time.time()
                    received += len(data)
                if received >= expected_bytes:
                    completed = time.time()
        except (TimeoutError, InterruptedError):
            pass
        finally:
            with open(result_path, "w") as output:
                json.dump(
                    {
                        "first_byte_at": first_byte,
                        "completed_at": completed,
                        "received_bytes": received,
                    },
                    output,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    sender = commands.add_parser("send")
    sender.add_argument("host")
    for name in ("port", "size_bytes", "rate_bps"):
        sender.add_argument(name, type=int)
    sender.add_argument("starts_at", type=float)
    sender.add_argument("timeout", type=float)
    sender.add_argument("result_path")
    receiver = commands.add_parser("receive")
    receiver.add_argument("port", type=int)
    receiver.add_argument("expected_bytes", type=int)
    receiver.add_argument("timeout", type=float)
    receiver.add_argument("result_path")
    replay = commands.add_parser("replay")
    replay.add_argument("starts_at", type=float)
    replay.add_argument("command", nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    operation = args.pop("operation")

    def finish_on_signal(signum: int, frame: FrameType | None) -> None:
        raise InterruptedError("Trial ended")

    if operation in ("send", "receive"):
        signal.signal(signal.SIGTERM, finish_on_signal)
    if operation == "send":
        send_tcp(**args)
    elif operation == "receive":
        receive_tcp(**args)
    else:
        time.sleep(max(0.0, args["starts_at"] - time.time()))
        os.execvp(args["command"][0], args["command"])


if __name__ == "__main__":
    main()
