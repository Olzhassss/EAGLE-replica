"""Compute per-slot reward components from acknowledged controller snapshots."""

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
