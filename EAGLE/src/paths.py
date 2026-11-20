"""Repository inputs and experiment outputs, independent of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
EVAL = ROOT / "eval"
DC_CAPTURE = ROOT / "5gb-tcp-connection.pcap.gz"
MG_CAPTURE = ROOT / "megalodon.pcapng"
DEFAULT_SETTINGS = ROOT / "settings/default.json"
SOURCE_NETWORK = Path(__file__).with_name("network_preparation") / "source_v0.json"
