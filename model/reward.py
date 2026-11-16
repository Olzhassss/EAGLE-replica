"""Compute per-slot reward components from acknowledged controller snapshots."""

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any


def compute_reward(
    snapshot: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    """Return the reward, delivered throughput in bps, and modeled watts."""
    duration = snapshot["sample_duration_seconds"]
    rates = [count * 8 / duration for count in snapshot["delivered_bytes"].values()]
    bandwidth = mean(rates) if rates else 0.0
    usage = snapshot["usage"]
    power = config["power_model"]
    watts = (
        usage["active_switches"] * power["idle_switch_watts"]
        + usage["active_ports"] * power["idle_port_watts"]
        + sum(usage["switch_egress_bps"].values())
        / 1_000_000
        * power["dynamic_watts_per_mbps"]
    )
    throughput = min(1.0, bandwidth / config["throughput_reference_bps"])
    cost = config["alpha"] * min(1.0, watts / config["power_reference_watts"])
    return {
        "reward": throughput - cost,
        "mean_delivered_flow_bps": bandwidth,
        "total_delivered_bps": sum(rates),
        "power_watts": watts,
        "throughput_term": throughput,
        "cost_term": cost,
    }


def compute_carbon_grams(
    sample_duration_seconds: float,
    dynamic_watts_by_node: Mapping[str, float],
    active_links: Iterable[tuple[str, str]],
    intensity_grams_per_kwh: Mapping[str, float],
    idle_port_watts: float,
) -> float:
    """Integrate the dynamic and active-port terms in study Eq. (35)."""
    dynamic = sum(
        watts * intensity_grams_per_kwh[node]
        for node, watts in dynamic_watts_by_node.items()
    )
    ports = idle_port_watts * sum(
        intensity_grams_per_kwh[u] + intensity_grams_per_kwh[v] for u, v in active_links
    )
    return (dynamic + ports) * sample_duration_seconds / 3_600_000
