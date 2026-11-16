"""Reduce saved EAGLE trials to reproducible comparisons and optional plots."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .run_log import read_events


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_run(directory: Path) -> dict[str, Any]:
    """Treat a run directory, including all its episodes, as one repetition."""
    manifest = json.loads((directory / "run.json").read_text())
    events = directory / "events.jsonl"
    steps = []
    flows = []
    learner = []
    returns = []
    errors = []
    completed_episodes = set()
    for row in read_events(events) if events.exists() else ():
        event = row["event"]
        if event == "transition":
            steps.append(row)
        elif event == "flow":
            flows.append(row)
        elif event == "learning":
            learner.append(row)
        elif event == "trial_end":
            completed_episodes.add(row["episode"])
            if "cumulative_reward" in row:
                returns.append(row["cumulative_reward"])
        elif event == "trial_error":
            errors.append(row["message"])
    if not steps:
        errors.append("No live transitions in run")
    if len(completed_episodes) != manifest["expected_episodes"]:
        errors.append(
            "Expected {} completed episodes, got {}".format(
                manifest["expected_episodes"],
                len(completed_episodes),
            ),
        )
    measured_seconds = sum(row["step_seconds"] for row in steps)
    available_power = bool(steps) and all(
        part["power_watts"] is not None
        for row in steps
        for part in row["reward_components"]
    )
    energy = (
        sum(
            row["step_seconds"]
            * sum(part["power_watts"] for part in row["reward_components"])
            for row in steps
        )
        if available_power
        else None
    )
    carbon = [
        row["carbon_grams"] for row in steps if row.get("carbon_grams") is not None
    ]
    complete = [
        row["fct_seconds"] for row in flows if row.get("fct_seconds") is not None
    ]
    matched_packets = sum(row.get("matched_packets", 0) for row in flows)
    dispatch = []
    barriers = []
    seen = set()
    for row in steps:
        for snapshot in row["snapshots"]:
            timing = snapshot["rule_timing"]
            key = (
                snapshot["served_by"],
                snapshot["incarnation"],
                timing.get("dispatched_at"),
            )
            if not timing or key in seen:
                continue
            seen.add(key)
            dispatch.append(timing["dispatch_seconds"])
            if "barrier_at" in timing:
                barriers.append(timing["barrier_at"] - timing["dispatched_at"])
    return {
        "directory": str(directory),
        "condition": manifest["condition"],
        "method": manifest["method"],
        "seed": manifest["seed"],
        "errors": errors,
        "fct_seconds": _mean(complete),
        "completed_flows": len(complete),
        "incomplete_flows": len(flows) - len(complete),
        "offered_bytes": sum(flow["offered_bytes"] for flow in flows),
        "received_bytes": sum(flow["received_bytes"] for flow in flows),
        "first_byte_seconds": _mean(
            [
                row["first_byte_seconds"]
                for row in flows
                if row.get("first_byte_seconds") is not None
            ],
        ),
        "matched_packets": matched_packets,
        "packet_delay_seconds": (
            sum(row.get("packet_delay_sum_seconds", 0) for row in flows)
            / matched_packets
            if matched_packets
            else None
        ),
        "throughput_bps": _mean(
            [
                sum(part["total_delivered_bps"] for part in row["reward_components"])
                for row in steps
            ],
        ),
        "power_watts": energy / measured_seconds
        if energy is not None and measured_seconds
        else None,
        "energy_joules": energy,
        "carbon_grams": sum(carbon) if carbon else None,
        "cumulative_reward": _mean(returns),
        "episode_returns": returns,
        "embedding_size": 4
        * manifest["encoder"]["order"]
        * manifest["encoder"]["eval_points"]
        * len(manifest["encoder"]["controller_ids"]),
        "step_reward": _mean([mean(row["reward"]) for row in steps]),
        "step_rewards": [mean(row["reward"]) for row in steps],
        "inference_seconds": _mean([row["inference_seconds"] for row in steps]),
        "rule_dispatch_seconds": _mean(dispatch),
        "openflow_barrier_seconds": _mean(barriers),
        "interface_seconds": _mean(
            [
                row["physical"]["interface_seconds"]
                for row in steps
                if row["physical"].get("interface_seconds") is not None
                and row["physical"].get("changes")
            ],
        ),
        "interface_on_seconds": _mean(_interface_timings(steps, enabled=True)),
        "interface_off_seconds": _mean(_interface_timings(steps, enabled=False)),
        "ovs_on_seconds": _mean(_ovs_timings(steps, enabled=True)),
        "ovs_off_seconds": _mean(_ovs_timings(steps, enabled=False)),
        "peer_parameter_bytes": sum(row["peer_parameter_bytes"] for row in learner)
        if learner
        else None,
        "worker_transport_bytes": sum(row["worker_transport_bytes"] for row in learner)
        if learner
        else None,
    }


def _ovs_timings(steps: list[dict[str, Any]], *, enabled: bool) -> list[float]:
    return [
        change["ovs_seconds"]
        for row in steps
        for change in row["physical"].get("changes", ())
        if change["enabled"] == enabled and change["ovs_seconds"] is not None
    ]


def _interface_timings(steps: list[dict[str, Any]], *, enabled: bool) -> list[float]:
    return [
        change["seconds"]
        for row in steps
        for change in row["physical"].get("interface_changes", ())
        if change["enabled"] == enabled
    ]


def _interval(
    values: list[float],
    rng: np.random.Generator,
    draws: int = 5000,
) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci90": None, "count": 0}
    ci = None
    if len(values) > 1:
        means = rng.choice(values, (draws, len(values))).mean(axis=1)
        ci = np.quantile(means, [0.05, 0.95]).tolist()
    return {"mean": mean(values), "ci90": ci, "count": len(values)}


def _relative(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return 1 - value / baseline


def compare_runs(
    runs: list[dict[str, Any]],
    baseline: str,
    seed: int,
    normalization: str,
) -> dict[str, Any]:
    """Pair each method with the same condition and seed in the static baseline."""
    groups = defaultdict(list)
    references = {}
    seen = set()
    for run in runs:
        condition, method, trial = (run[key] for key in ("condition", "method", "seed"))
        key = (condition, method, trial)
        if key in seen:
            raise ValueError(f"Duplicate condition/method/seed: {key!r}")
        seen.add(key)
        if not run["errors"]:
            groups[condition, method].append((trial, run))
            if method == baseline:
                references[condition, trial] = run
    deltas = {}
    maxima = defaultdict(list)
    for (condition, method), group in groups.items():
        pairs = []
        for trial, run in group:
            reference = references.get((condition, trial))
            if reference is None:
                continue
            gain = _relative(run["fct_seconds"], reference["fct_seconds"])
            power = _relative(run["power_watts"], reference["power_watts"])
            pairs.append((gain, power))
        gains = [gain for gain, _ in pairs if gain is not None]
        if gains:
            maxima[condition if normalization == "condition" else "all"].append(
                (condition, method, mean(gains)),
            )
        deltas[condition, method] = pairs

    denominators = {
        key: {"largest_mean_fct_gain": max(row[2] for row in rows), "candidates": rows}
        for key, rows in maxima.items()
    }
    rng = np.random.default_rng(seed)
    comparisons = []
    metrics = (
        "fct_seconds",
        "power_watts",
        "carbon_grams",
        "cumulative_reward",
        "step_reward",
        "first_byte_seconds",
        "packet_delay_seconds",
        "throughput_bps",
        "inference_seconds",
        "rule_dispatch_seconds",
        "openflow_barrier_seconds",
        "interface_seconds",
        "interface_on_seconds",
        "interface_off_seconds",
        "ovs_on_seconds",
        "ovs_off_seconds",
        "peer_parameter_bytes",
        "worker_transport_bytes",
    )
    for condition, method in sorted(groups):
        entries = [run for _, run in groups[condition, method]]
        values = {
            name: [run[name] for run in entries if run[name] is not None]
            for name in metrics
        }
        pairs = deltas[condition, method]
        maximum = denominators.get(
            condition if normalization == "condition" else "all",
            {},
        )
        scale = maximum.get("largest_mean_fct_gain", 0)
        values["fct_gain"] = [gain for gain, _ in pairs if gain is not None]
        values["power_saving"] = [power for _, power in pairs if power is not None]
        values["efficiency"] = [
            gain / scale / (1 - power)
            for gain, power in pairs
            if scale > 0 and gain is not None and power is not None and power < 1
        ]
        comparisons.append(
            {
                "condition": condition,
                "method": method,
                "runs": len(entries),
                "matched_baseline_runs": len(pairs),
                "metrics": {
                    name: _interval(samples, rng) for name, samples in values.items()
                },
            },
        )
    return {
        "baseline": baseline,
        "interval": "90% percentile bootstrap by run",
        "bootstrap_seed": seed,
        "efficiency_normalization": normalization,
        "fct_gain_denominators": denominators,
        "runs": runs,
        "comparisons": comparisons,
    }


def plot_comparisons(report: dict[str, Any], directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for metric in (
        "fct_gain",
        "power_saving",
        "efficiency",
        "carbon_grams",
        "step_reward",
        "packet_delay_seconds",
        "inference_seconds",
        "rule_dispatch_seconds",
        "interface_on_seconds",
        "interface_off_seconds",
        "ovs_on_seconds",
        "ovs_off_seconds",
        "peer_parameter_bytes",
        "cumulative_reward",
    ):
        series = [
            (row["condition"] + " / " + row["method"], row["metrics"][metric])
            for row in report["comparisons"]
            if row["metrics"][metric]["mean"] is not None
        ]
        if not series:
            continue
        figure, axis = plt.subplots(
            figsize=(max(7, len(series)), 4),
            layout="constrained",
        )
        axis.bar(range(len(series)), [stat["mean"] for _, stat in series])
        for position, (_, stat) in enumerate(series):
            if stat["ci90"] is not None:
                axis.vlines(position, *stat["ci90"], color="black")
        axis.set_xticks(
            range(len(series)),
            [name for name, _ in series],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel(metric.replace("_", " "))
        figure.savefig(directory / (metric + ".png"))
        plt.close(figure)

    learning_curves = defaultdict(list)
    for run in report["runs"]:
        if not run["errors"] and run["episode_returns"]:
            learning_curves[run["condition"], run["method"]].append(
                run["episode_returns"],
            )
    if learning_curves:
        figure, axis = plt.subplots(layout="constrained")
        for (condition, method), series in sorted(learning_curves.items()):
            points = [mean(point) for point in zip(*series)]
            axis.plot(
                range(1, len(points) + 1),
                points,
                label=condition + " / " + method,
            )
        axis.set(xlabel="Episode", ylabel="Cumulative reward")
        axis.legend()
        figure.savefig(directory / "episode_returns.png")
        plt.close(figure)


def plot_embeddings(runs: list[dict[str, Any]], output: Path, seed: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    samples = defaultdict(list)
    for run in runs:
        if run["errors"]:
            continue
        for row in read_events(Path(run["directory"]) / "events.jsonl"):
            if row["event"] in ("reset", "transition"):
                samples[len(row["raw_embedding"])].append(
                    (run["condition"], row["raw_embedding"]),
                )
    for width, points in samples.items():
        if len(points) < 3:
            continue
        labels, vectors = zip(*points)
        projection = TSNE(
            n_components=2,
            perplexity=min(30, len(points) - 1),
            init="random",
            random_state=seed,
        ).fit_transform(np.asarray(vectors))
        rows = [
            {"condition": name, "x": float(point[0]), "y": float(point[1])}
            for name, point in zip(labels, projection)
        ]
        target = output.with_name(output.stem + "-" + str(width))
        target.with_suffix(".json").write_text(
            json.dumps({"seed": seed, "points": rows}, indent=2) + "\n",
        )
        figure, axis = plt.subplots(layout="constrained")
        for name in sorted(set(labels)):
            subset = projection[np.asarray(labels) == name]
            axis.scatter(subset[:, 0], subset[:, 1], label=name, s=14)
        axis.legend()
        figure.savefig(target.with_suffix(".png"))
        plt.close(figure)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--baseline", default="static")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path)
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Path prefix for t-SNE data and plots by feature width",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--efficiency-normalization",
        choices=("condition", "global"),
        default="condition",
    )
    args = parser.parse_args()
    try:
        runs = [summarize_run(path) for path in args.runs]
        report = compare_runs(
            runs,
            args.baseline,
            args.bootstrap_seed,
            args.efficiency_normalization,
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.plots:
            plot_comparisons(report, args.plots)
        if args.embeddings:
            plot_embeddings(runs, args.embeddings, args.bootstrap_seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    _main()
