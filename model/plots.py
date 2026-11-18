"""Matched-baseline metrics and figures for the five evaluation workflows."""

from pathlib import Path
from statistics import mean

import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from .artifacts import read_rewards, write_json

METRICS = {
    "fct_gain": "FCT gain", "power_saving": "Power saving (%)", "efficiency": "Efficiency score",
}
DATASET_LABELS = {
    "dc_t0": "DC synthetic T0", "dc_pcap": "DC PCAPs",
    "mg_t0": "MG synthetic T2", "mg_pcap": "MG PCAPs",
    "dc_t1": "T1", "dc_t3": "T3", "dc_t4": "T4",
    **{f"dc_t0_trenv{index}": f"TrEnv {index}" for index in range(2, 6)},
}
COLORS = ("#3374a5", "#e49c39", "#45966c", "#bd647c", "#7a68a8")


def interval(values: list[float | None], rng: np.random.Generator) -> dict:
    samples = np.array([value for value in values if value is not None], dtype=float)
    if not len(samples):
        return {"mean": None, "ci90": None}
    bounds = np.quantile(rng.choice(samples, (5000, len(samples))).mean(axis=1), [0.05, 0.95])
    return {"mean": float(samples.mean()), "ci90": bounds.tolist()}


def compare_condition(condition: dict, models: list[str], seed: int) -> dict:
    """Normalize mean FCT gain across models within this dataset and topology."""
    baseline = condition["measurements"]["baseline"]
    values = {}
    for name in models:
        gains, savings = [], []
        for trial, reference in zip(condition["measurements"][name], baseline, strict=True):
            if trial["seed"] != reference["seed"]:
                raise ValueError("A comparison requires matching repetition seeds")
            gains.append(relative_gain(trial["fct_seconds"], reference["fct_seconds"]))
            savings.append(relative_gain(trial["power_watts"], reference["power_watts"]))
        values[name] = {"fct_gain": gains, "power_saving": savings}
    averages = []
    for metrics in values.values():
        available = [value for value in metrics["fct_gain"] if value is not None]
        if available:
            averages.append(mean(available))
    maximum = max(averages, default=0)
    rng = np.random.default_rng(seed)
    summaries = {}
    for name, metrics in values.items():
        metrics["efficiency"] = [
            gain / maximum / (1 - saving)
            if gain is not None and saving is not None and maximum > 0 and saving < 1
            else None
            for gain, saving in zip(metrics["fct_gain"], metrics["power_saving"], strict=True)
        ]
        summaries[name] = {metric: interval(samples, rng) for metric, samples in metrics.items()}
    return dict(condition, largest_mean_fct_gain=maximum, metrics=summaries)


def relative_gain(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    return 1 - value / reference


def plot_bars(
    conditions: list[dict], models: list[str], target: Path, *, topologies: bool = False,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = [condition["network"] if topologies else DATASET_LABELS[condition["dataset"]] for condition in conditions]
    for metric, ylabel in METRICS.items():
        figure = Figure(figsize=(max(8, len(labels) * 1.7), 4.5), layout="constrained")
        axis = figure.subplots()
        positions = np.arange(len(conditions))
        width = 0.23
        scale = 100 if metric == "power_saving" else 1
        for index, name in enumerate(models):
            summaries = [condition["metrics"][name][metric] for condition in conditions]
            heights = [row["mean"] * scale if row["mean"] is not None else np.nan for row in summaries]
            errors = np.array([
                [max(0, row["mean"] - row["ci90"][0]) * scale,
                 max(0, row["ci90"][1] - row["mean"]) * scale]
                if row["mean"] is not None else [0, 0]
                for row in summaries
            ]).T
            axis.bar(
                positions + (index - 1) * width, heights, width, yerr=errors,
                capsize=3, color=COLORS[index], label=name,
            )
        axis.set_xticks(positions, labels)
        axis.set(ylabel=ylabel)
        axis.axhline(0, color="#65717a", linewidth=0.6)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        figure.savefig(target.with_name(target.name + "_" + metric + ".png"), dpi=170)


def plot_embeddings(samples: list[dict], target: Path, seed: int) -> None:
    vectors = np.asarray([sample["embedding"] for sample in samples], dtype=float)
    if vectors.ndim != 2 or len(vectors) < 3 or not np.isfinite(vectors).all():
        raise ValueError("t-SNE requires at least three finite embeddings of equal width")
    projection = TSNE(
        n_components=2, perplexity=min(30, max(1, (len(vectors) - 1) / 3)),
        init="random", random_state=seed,
    ).fit_transform(vectors)
    figure = Figure(figsize=(8, 6), layout="constrained")
    axis = figure.subplots()
    labels = list(dict.fromkeys(sample["label"] for sample in samples))
    for index, label in enumerate(labels):
        selected = np.array([sample["label"] == label for sample in samples])
        points = projection[selected]
        axis.scatter(
            points[:, 0], points[:, 1], label=label, color=COLORS[index % len(COLORS)],
            marker=("o", "s", "^", "D", "v")[index // len(COLORS)], s=36, alpha=0.85,
        )
    axis.set(xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    axis.legend(loc="best", fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target.with_name(target.name + ".png"), dpi=170)
    write_json(target.with_name(target.name + "_projection.json"), {
        "seed": seed,
        "points": [
            dict(sample, x=float(point[0]), y=float(point[1]))
            for sample, point in zip(samples, projection, strict=True)
        ],
    })


def plot_pruning(paths: list[Path], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    curves = [read_rewards(path) for path in paths]
    baseline_episodes, baseline = curves[1]
    for relative in (False, True):
        figure = Figure(figsize=(8, 4.5), layout="constrained")
        axis = figure.subplots()
        labels = ("Dynamic pruning", "No pruning", "Static pruning")
        for index, (label, (episodes, rewards)) in enumerate(zip(labels, curves, strict=True)):
            if relative:
                common, current_indices, baseline_indices = np.intersect1d(episodes, baseline_episodes, return_indices=True)
                reference = baseline[baseline_indices]
                values = np.divide(
                    100 * (rewards[current_indices] - reference), np.abs(reference),
                    out=np.full(len(common), np.nan), where=reference != 0,
                )
                episodes, rewards = common, values
            axis.plot(episodes, rewards, color=COLORS[index], label=label)
        axis.set(xlabel="Episode", ylabel="Reward relative to no pruning (%)" if relative else "Cumulative reward")
        axis.grid(alpha=0.25)
        axis.legend()
        suffix = "relative" if relative else "reward"
        figure.savefig(target / f"pruning_{suffix}.png", dpi=170)
