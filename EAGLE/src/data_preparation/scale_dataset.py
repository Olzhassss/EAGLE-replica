import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from data_preparation.traffic_loads import scale_synthetic

SCALE_FACTOR = 0.2


def main() -> None:
    directory = ROOT / "datasets" / "9_hosts" / "train"
    print(directory / "dc_t5.json")
    scale_synthetic(directory / "dc_t0.json", directory / "dc_t5.json", SCALE_FACTOR)


if __name__ == "__main__":
    main()
