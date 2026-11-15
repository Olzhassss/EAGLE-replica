"""Compute per-slot reward components from acknowledged controller snapshots."""

from statistics import mean


def compute_reward(snapshot, config):
    """Return the reward, delivered throughput in bps, and optional modeled watts."""
    duration = snapshot["sample_duration_seconds"]
    rates = [count * 8 / duration for count in snapshot["delivered_bytes"].values()]
    bandwidth = mean(rates) if rates else 0.0
    usage = snapshot["usage"]
    power = config.get("power_model")
    watts = None
    if power is not None:
        watts = (
            usage["active_switches"] * power["idle_switch_watts"]
            + usage["active_ports"] * power["idle_port_watts"]
            + sum(usage["switch_egress_bps"].values()) / 1_000_000 * power["dynamic_watts_per_mbps"]
        )
    if config["kind"] == "legacy":
        samples = [value for value in snapshot["throughput"].values() if value]
        slot = snapshot["controller_id"]
        active = snapshot["resources"][slot]["action"]
        throughput = mean(samples) * 8 / 1024 / 1024 / config["throughput_scale"] if samples else 0.0
        cost = active * config["overprovisioning_penalty"] / 100
        reward = int((throughput - cost) * 100)
    else:
        throughput = min(1.0, bandwidth / config["throughput_reference_bps"])
        cost = config["alpha"] * min(1.0, watts / config["power_reference_watts"])
        reward = throughput - cost
    return {
        "reward": reward, "mean_delivered_flow_bps": bandwidth, "total_delivered_bps": sum(rates),
        "power_watts": watts, "throughput_term": throughput, "cost_term": cost,
    }


def compute_carbon_grams(sample_duration_seconds, dynamic_watts_by_node, active_links,
                         intensity_grams_per_kwh, idle_port_watts):
    """Integrate the dynamic and active-port terms in study Eq. (35)."""
    dynamic = sum(watts * intensity_grams_per_kwh[node] for node, watts in dynamic_watts_by_node.items())
    ports = idle_port_watts * sum(intensity_grams_per_kwh[u] + intensity_grams_per_kwh[v] for u, v in active_links)
    return (dynamic + ports) * sample_duration_seconds / 3_600_000
