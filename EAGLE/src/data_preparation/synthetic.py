"""Write synthetic traffic profiles, their scaled variants and throughput plots."""

import argparse
import json
import math
from pathlib import Path

from EAGLE.src.data_preparation.hosts import Host
from EAGLE.src.data_preparation.windows import Window


LOAD_FACTORS = {1: 0.5, 3: 1.5, 4: 2.0}


def write_chart(
    path: Path,
    rates: dict[tuple[str, str], list[float]],
    *,
    window_seconds: float = 1,
) -> None:
    """Plot total Mbps and each directed pair over the prepared time grid."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directions = list(rates)
    values = list(rates.values())
    steps = len(values[0])
    duration = steps * window_seconds
    figure = Figure(
        figsize=(min(24, max(9, steps * 0.35)), 3 + 0.19 * len(directions)),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    total, pairs = figure.subplots(
        2, 1, sharex=True, height_ratios=[2, max(2, len(directions) / 5)]
    )
    edges = [index * window_seconds for index in range(steps + 1)]
    total.stairs(list(map(sum, zip(*values, strict=True))), edges)
    total.set(ylabel="Total Mbps", title=path.stem, ylim=(0, None))
    total.grid(axis="y", alpha=0.25)
    image = pairs.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=max(map(max, values)) or 1,
        extent=(0, duration, len(directions) - 0.5, -0.5),
    )
    pairs.set_yticks(
        range(len(directions)), [f"{a} → {b}" for a, b in directions], fontsize=8
    )
    pairs.set(xlabel="Elapsed seconds", xlim=(0, duration))
    figure.colorbar(image, ax=[total, pairs], label="Mbps per direction")
    figure.savefig(path.with_suffix(".png"), dpi=120)


def _write_synthetic(path: Path, series: list[dict], metadata: dict) -> None:
    """Convert directed byte-volume windows to scheduled TCP jobs and a traffic chart."""
    jobs = []
    rates = {}
    window_seconds = metadata.get("window_seconds", 1)
    for index, direction in enumerate(series):
        source, target = direction["source"], direction["target"]
        throughputs = []
        for elapsed, window in enumerate(direction["windows"]):
            byte_count = window["frame_bytes"]
            window["average_mbps"] = byte_count * 8 / window_seconds / 1_000_000
            throughputs.append(window["average_mbps"])
            if byte_count:
                size = max(1, round(byte_count))
                jobs.append(
                    {
                        "id": f"flow-{index + 1}-{elapsed}",
                        "kind": "tcp",
                        "source": source,
                        "target": target,
                        "start_seconds": elapsed * window_seconds,
                        "size_bytes": size,
                        "rate_bps": size * 8 / window_seconds,
                        "timeout_seconds": max(60, math.ceil(window_seconds) + 5),
                    }
                )
        rates[source, target] = throughputs
    profile = {
        **metadata,
        "duration_seconds": len(series[0]["windows"]) * window_seconds,
        "series": series,
        "jobs": jobs,
    }
    path.write_text(json.dumps(profile, indent=2) + "\n")
    write_chart(path, rates, window_seconds=window_seconds)


def generate_synthetic_dataset(
    path: Path,
    pools: list[list[Window]],
    pairs: list[tuple[Host, Host]],
    *,
    time_scale: float = 1,
) -> None:
    """Preserve captured byte volumes while stretching the delivery interval."""
    series = [
        {
            "source": source.name,
            "target": target.name,
            "windows": [
                {
                    "capture_second": window.second,
                    "capture_phase": window.phase,
                    "packets_per_second": window.packets / time_scale,
                    "frame_bytes": window.frame_bytes,
                }
                for window in windows
            ],
        }
        for (source, target), windows in zip(pairs, pools, strict=True)
    ]
    hosts = {host.name: host.to_dict() for pair in pairs for host in pair}
    _write_synthetic(
        path,
        series,
        {
            "schema_version": 1,
            "hosts": list(hosts.values()),
            "traffic_scale_factor": time_scale,
            "window_seconds": time_scale,
        },
    )


def scale_synthetic(
    source: Path, target: Path, factor: float, sources: set[str] | None = None
) -> None:
    """Scale selected source volumes and regenerate the manifest jobs and chart."""
    data = json.loads(source.read_text())
    series = data.pop("series")
    data.pop("jobs", None)
    for direction in series:
        if sources is None or direction["source"] in sources:
            for window in direction["windows"]:
                window["frame_bytes"] *= factor
                window["packets_per_second"] *= factor
    _write_synthetic(target, series, data)


def main() -> None:
    """Scale a synthetic manifest and regenerate its jobs and throughput chart."""
    parser = argparse.ArgumentParser(description="Scale a synthetic traffic dataset")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--sources", nargs="+", help="Scale only these source host IDs")
    args = parser.parse_args()
    if not math.isfinite(args.factor) or args.factor < 0:
        parser.error("--factor must be finite and nonnegative")
    scale_synthetic(
        args.source,
        args.target,
        args.factor,
        set(args.sources) if args.sources else None,
    )


if __name__ == "__main__":
    main()
