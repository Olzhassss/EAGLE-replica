"""Launch a generated network and serve prepared traffic jobs."""

import argparse
import hashlib
import json
import os
import signal
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

from mininet.control_server import ControlRequest, NetworkSession
from mininet.network import populate_network, validate_network
from mininet.traffic import TrialTraffic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Start Mininet and its owner-only control socket, then clean up on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=ROOT / "networks/v0.json")
    parser.add_argument(
        "--cli", action="store_true", help="Open an interactive Mininet CLI"
    )
    args = parser.parse_args()
    if os.geteuid() != 0 or sys.prefix != sys.base_prefix:
        parser.error("Use sudo python3 with the system Mininet installation")
    from mininet.cli import CLI
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch, RemoteController

    raw = args.network.read_bytes()
    network = json.loads(raw)
    validate_network(network)
    fingerprint = hashlib.sha256(raw).hexdigest()
    owner_uid = int(os.environ.get("SUDO_UID", "0"))
    owner_gid = int(os.environ.get("SUDO_GID", "0"))
    mininet = Mininet(
        topo=None, controller=None, switch=OVSKernelSwitch, link=TCLink, build=False
    )

    def stop_on_signal(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    with tempfile.TemporaryDirectory(prefix="eagle-mininet-") as directory:
        session_dir = Path(directory)
        session_dir.chmod(0o711)
        traffic = TrialTraffic(mininet, session_dir, owner_uid)
        try:
            populate_network(mininet, network, RemoteController)
            mininet.start()
            mininet.staticArp()
            address = str(session_dir / "control.sock")
            with socketserver.UnixStreamServer(address, ControlRequest) as server:
                server.owner_uid = owner_uid
                server.session = NetworkSession(fingerprint, traffic)
                os.chown(address, owner_uid, owner_gid)
                os.chmod(address, 0o600)
                socket_path = Path(__file__).resolve().with_name(".socket")
                address_file = socket_path.open("x", encoding="utf-8")
                try:
                    with address_file:
                        address_file.write(address)
                    if not args.cli:
                        server.serve_forever(poll_interval=0.25)
                    else:
                        thread = threading.Thread(
                            target=server.serve_forever, daemon=True
                        )
                        thread.start()
                        try:
                            CLI(mininet)
                        finally:
                            server.shutdown()
                            thread.join()
                finally:
                    socket_path.unlink(missing_ok=True)
        except KeyboardInterrupt:
            pass
        finally:
            traffic.finish()
            mininet.stop()


if __name__ == "__main__":
    main()
