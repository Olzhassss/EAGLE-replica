"""Compare saved training reward curves for dynamic, absent, and static pruning."""

import argparse

from .artifacts import EVAL, MODELS
from .plots import plot_pruning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    try:
        plot_pruning([MODELS / f"{name}.csv" for name in ("dc_t0", "dc_t0_no_pruning", "dc_t0_static_pruning")], EVAL)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
