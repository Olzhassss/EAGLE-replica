"""Write synthetic traffic profiles, their scaled variants and throughput plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .load_and_split import Window

if TYPE_CHECKING:
    from .replay import Host

LOAD_FACTORS = {1: 0.5, 3: 1.5, 4: 2.0}


def write_chart(path: Path, rates: dict[tuple[str, str], list[float]]) -> None:
    """Plot total Mbps and each directed pair on the same one-second grid."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directions = list(rates)
    values = list(rates.values())
    steps = len(values[0])
    figure = Figure(
        figsize=(min(24, max(9, steps * 0.35)), 3 + 0.19 * len(directions)),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    total, pairs = figure.subplots(
        2, 1, sharex=True, height_ratios=[2, max(2, len(directions) / 5)]
    )
    total.stairs(list(map(sum, zip(*values, strict=True))), range(steps + 1))
    total.set(ylabel="Total Mbps", title=path.stem, ylim=(0, None))
    total.grid(axis="y", alpha=0.25)
    image = pairs.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=max(map(max, values)) or 1,
        extent=(0, steps, len(directions) - 0.5, -0.5),
    )
    pairs.set_yticks(
        range(len(directions)), [f"{a} → {b}" for a, b in directions], fontsize=8
    )
    pairs.set(xlabel="Elapsed seconds", xlim=(0, steps))
    figure.colorbar(image, ax=[total, pairs], label="Mbps per direction")
    figure.savefig(path.with_suffix(".png"), dpi=120)


def _write_synthetic(path: Path, series: list[dict], metadata: dict) -> None:
    jobs = []
    rates = {}
    for index, direction in enumerate(series):
        source, target = direction["source"], direction["target"]
        throughputs = []
        for elapsed, window in enumerate(direction["windows"]):
            byte_count = window["frame_bytes"]
            window["average_mbps"] = byte_count * 8 / 1_000_000
            throughputs.append(window["average_mbps"])
            if byte_count:
                size = max(1, round(byte_count))
                jobs.append(
                    {
                        "id": f"flow-{index + 1}-{elapsed}",
                        "kind": "tcp",
                        "source": source,
                        "target": target,
                        "start_seconds": elapsed,
                        "size_bytes": size,
                        "rate_bps": size * 8,
                    }
                )
        rates[source, target] = throughputs
    profile = {
        **metadata,
        "duration_seconds": len(series[0]["windows"]),
        "series": series,
        "jobs": jobs,
    }
    path.write_text(json.dumps(profile, indent=2) + "\n")
    write_chart(path, rates)


def generate_synthetic_dataset(
    path: Path, pools: list[list[Window]], pairs: list[tuple[Host, Host]]
) -> None:
    series = [
        {
            "source": source.name,
            "target": target.name,
            "windows": [
                {
                    "capture_second": window.second,
                    "capture_phase": window.phase,
                    "packets_per_second": window.packets,
                    "frame_bytes": window.frame_bytes,
                }
                for window in windows
            ],
        }
        for (source, target), windows in zip(pairs, pools, strict=True)
    ]
    hosts = {host.name: host.to_dict() for pair in pairs for host in pair}
    _write_synthetic(path, series, {"schema_version": 1, "hosts": list(hosts.values())})


def scale_synthetic(
    source: Path, target: Path, factor: float, sources: set[str] | None = None
) -> None:
    data = json.loads(source.read_text())
    series = data.pop("series")
    data.pop("jobs", None)
    for direction in series:
        if sources is None or direction["source"] in sources:
            for window in direction["windows"]:
                window["frame_bytes"] *= factor
                window["packets_per_second"] *= factor
    _write_synthetic(target, series, data)


def generate_loads(test: Path) -> None:
    for pattern, factor in LOAD_FACTORS.items():
        scale_synthetic(test / "dc_t0.json", test / f"dc_t{pattern}.json", factor)
