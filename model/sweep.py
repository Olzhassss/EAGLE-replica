"""Schedule named evaluation cells using existing controller and Mininet sessions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .settings import ROOT


def run_matrix(manifest: dict[str, Any], directory: Path) -> bool:
    """Run the same seeds in every cell and retain exit codes even on failure."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected a schema_version 1 evaluation matrix")
    seeds = manifest.get("seeds", list(range(10)))
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Use distinct nonnegative matrix seeds")
    cells = manifest["cells"]
    identifiers = set()
    protected = {"--seed", "--output", "--condition", "--method"}
    for cell in cells:
        condition, method = cell["condition"], cell["method"]
        if any(
            not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", "..")
            for value in (condition, method)
        ):
            raise ValueError("Use path-safe condition and method names")
        if (condition, method) in identifiers:
            raise ValueError("Each condition/method pair may appear once")
        identifiers.add((condition, method))
        if any(flag.split("=", 1)[0] in protected for flag in cell["arguments"]):
            raise ValueError("The matrix sets seed, output, condition, and method")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failed = False
    with (directory / "outcomes.jsonl").open("a", buffering=1) as outcomes:
        for cell in cells:
            for seed in seeds:
                condition, method = cell["condition"], cell["method"]
                output = directory / condition / method / str(seed)
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "model.evaluate",
                    *cell["arguments"],
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                    "--condition",
                    condition,
                    "--method",
                    method,
                ]
                with output.with_suffix(".log").open("w") as log:
                    status = subprocess.run(  # noqa: S603
                        command,
                        check=False,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ).returncode
                outcomes.write(
                    json.dumps(
                        {
                            "condition": condition,
                            "method": method,
                            "seed": seed,
                            "run": str(output),
                            "exit_code": status,
                        },
                    )
                    + "\n",
                )
                failed |= bool(status)
    return failed


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        failed = run_matrix(json.loads(args.matrix.read_text()), args.output.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    raise SystemExit(int(failed))


if __name__ == "__main__":
    _main()
