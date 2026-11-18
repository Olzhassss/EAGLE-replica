"""Run scheduled TCP jobs with one asynchronous sender and receiver per host."""

import argparse
import asyncio
import json
import os
import re
import signal
import time
from pathlib import Path

PORT = 20000


async def before_deadline(awaitable, deadline: float):
    return await asyncio.wait_for(awaitable, timeout=max(0.0, deadline - time.monotonic()))


async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, directory: Path) -> None:
    job_id = None
    received = 0
    first = completed = None
    try:
        header = await asyncio.wait_for(reader.readline(), timeout=5)
        job = json.loads(header)
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]):
            return
        job_id = job["id"]
        deadline = time.monotonic() + job["timeout_seconds"]
        while received < job["size_bytes"]:
            data = await before_deadline(
                reader.read(min(65536, job["size_bytes"] - received)), deadline,
            )
            if not data:
                break
            first = time.time() if first is None else first
            received += len(data)
        if received >= job["size_bytes"]:
            completed = time.time()
    except (OSError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
        pass
    finally:
        writer.close()
        if job_id is not None and job.get("measure_flows", True):
            (directory / (job_id + ".received.json")).write_text(json.dumps({
                "first_byte_at": first, "completed_at": completed, "received_bytes": received,
            }))


async def serve(directory: Path) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: receive(reader, writer, directory),
        "0.0.0.0", PORT, backlog=512, limit=4096,
    )
    async with server:
        (directory / "ready").touch()
        await server.serve_forever()


async def send(job: dict, directory: Path) -> None:
    await asyncio.sleep(max(0.0, job["scheduled_at"] - time.time()))
    started = time.time()
    sent = 0
    error = None
    writer = None
    try:
        deadline = time.monotonic() + job["timeout_seconds"]
        _, writer = await before_deadline(asyncio.open_connection(job["address"], PORT), deadline)
        header = {key: job[key] for key in ("id", "size_bytes", "timeout_seconds")}
        header["measure_flows"] = job.get("measure_flows", True)
        writer.write(json.dumps(header).encode() + b"\n")
        await before_deadline(writer.drain(), deadline)
        payload = bytes(65536)
        while sent < job["size_bytes"]:
            block = payload[:min(len(payload), job["size_bytes"] - sent)]
            writer.write(block)
            await before_deadline(writer.drain(), deadline)
            sent += len(block)
            delay = started + sent * 8 / job["rate_bps"] - time.time()
            if delay > 0 and sent < job["size_bytes"]:
                await before_deadline(asyncio.sleep(delay), deadline)
    except (OSError, asyncio.TimeoutError) as failure:
        error = str(failure)
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=5)
            except (OSError, asyncio.TimeoutError):
                pass
        if job.get("measure_flows", True):
            (directory / (job["id"] + ".sent.json")).write_text(json.dumps({
                "started_at": started, "sent_bytes": sent, "error": error,
            }))


async def batch(path: Path, directory: Path) -> None:
    jobs = json.loads(path.read_text())
    await asyncio.gather(*(send(job, directory) for job in jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    server = commands.add_parser("serve")
    server.add_argument("directory", type=Path)
    sender = commands.add_parser("batch")
    sender.add_argument("path", type=Path)
    sender.add_argument("directory", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("starts_at", type=float)
    replay.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    def stop(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        if args.operation == "serve":
            asyncio.run(serve(args.directory))
        elif args.operation == "batch":
            asyncio.run(batch(args.path, args.directory))
        else:
            time.sleep(max(0.0, args.starts_at - time.time()))
            os.execvp(args.command[0], args.command)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
