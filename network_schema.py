"""Network support-switch and action contracts shared by all three runtimes.

Keep this module dependency-free and Python 3.7 compatible. Bit i of a slot's
integer action enables support_switches[i]; zero disables every support switch.
"""

from typing import Any, Dict, Sequence, Tuple, Union

NETWORK_SCHEMA_VERSION = 2


def resolve_support_counts(
    value: Union[int, Sequence[int]], domains: int
) -> Tuple[int, ...]:
    """Expand a uniform count or validate counts in controller order.

    The 62-switch bound keeps categorical action sizes inside signed int64.
    Actors have 2**count outputs, so practical configurations use small counts.
    """
    try:
        counts = (value,) * domains if type(value) is int else tuple(value)
    except TypeError as error:
        raise ValueError("Support-switch counts must be integers") from error
    if len(counts) == 1:
        counts *= domains
    if len(counts) != domains or any(
        type(count) is not int or not 0 <= count <= 62 for count in counts
    ):
        raise ValueError("Use one support-switch count or one per domain, from 0 to 62")
    return counts


def valid_support_action(controller: Dict[str, Any], action: object) -> bool:
    """Accept only integer bitmasks within a controller's configured action space."""
    return type(action) is int and 0 <= action < (
        1 << len(controller["support_switches"])
    )


def validate_support_layout(network: Dict[str, Any]) -> None:
    """Validate a versioned network's slot ownership, support lists, and initial masks."""
    if network.get("schema_version") != NETWORK_SCHEMA_VERSION:
        raise ValueError("Expected network schema version 2; regenerate older networks")
    controllers = network["controllers"]
    switches = {row["id"]: row["controller"] for row in network["switches"]}
    slots = {row["id"] for row in controllers}
    if not controllers or len(slots) != len(controllers):
        raise ValueError("Network controller IDs must be nonempty and unique")
    if len(switches) != len(network["switches"]) or set(switches.values()) - slots:
        raise ValueError("Switches need unique IDs and configured controller owners")
    supports = set()
    for controller in controllers:
        nodes = controller.get("support_switches")
        if not isinstance(nodes, list) or not all(
            isinstance(node, str) for node in nodes
        ):
            raise ValueError("Each controller needs an ordered support_switches list")
        resolve_support_counts(len(nodes), 1)
        if len(set(nodes)) != len(nodes) or any(
            switches.get(node) != controller["id"] for node in nodes
        ):
            raise ValueError(
                "Support switches must be distinct and belong to their domain"
            )
        if not valid_support_action(controller, controller["initial_action"]):
            raise ValueError("initial_action must be a valid support-switch bitmask")
        supports.update(nodes)
    hosts = {row["id"] for row in network["hosts"]}
    for link in network["links"]:
        a, b = link["source"], link["target"]
        if (a in supports and b in hosts) or (b in supports and a in hosts):
            raise ValueError(
                "Hosts must attach to fixed switches, outside support_switches"
            )
