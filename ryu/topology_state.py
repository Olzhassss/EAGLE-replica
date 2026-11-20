"""Track forwarding state and logical slot ownership for a configured network."""

from typing import Any, Dict, List, Tuple, Union

import networkx as nx
from network_schema import valid_support_action

from ryu.network import Network


class TopologyState:
    """Track acknowledged switch actions, elected slot owners, and the active graph."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self.configured = network.graph
        self.resources = {
            slot: {"action": controller["initial_action"], "revision": [0, ""]}
            for slot, controller in network.controllers.items()
        }
        self.active = nx.Graph()
        self.owners = {}
        self.connected = set()

    def merge_resources(self, presence: Dict[str, Dict[str, Any]]) -> None:
        """Accept the latest acknowledged action for each logical controller slot."""
        for slot, update in presence.items():
            current = self.resources.get(slot)
            if (
                current is not None
                and valid_support_action(self.network.controllers[slot], update["action"])
                and update["revision"] > current["revision"]
            ):
                self.resources[slot] = update

    def set_support_action(
        self,
        slot: str,
        action: int,
        revision: List[Union[int, str]],
    ) -> Dict[str, Any]:
        """Advance an action revision only when the domain's support mask changes."""
        if not valid_support_action(self.network.controllers[slot], action):
            raise ValueError("Action is outside the domain's support-switch space")
        current = self.resources[slot]
        if current["action"] != action:
            self.resources[slot] = {"action": action, "revision": revision}
        return self.resources[slot]

    def update_presence(self, frames: Dict[str, Dict[str, Any]]) -> None:
        """Elect physical controllers and rebuild the forwarding graph.

        All switches connect to each configured controller. A live primary keeps
        its logical slot. A failed primary's slot moves to the first live peer
        in network order. A switch is reachable through its elected owner.
        """
        live = [slot for slot in self.network.controllers if slot in frames]
        self.owners = (
            {slot: slot if slot in frames else live[0] for slot in self.resources}
            if live
            else {}
        )
        self.connected = set()
        for switch, row in self.network.switches.items():
            frame = frames.get(self.owners.get(row["controller"]), {})
            if switch in frame.get("ports", {}):
                self.connected.add(switch)
        disabled = {
            node
            for slot, update in self.resources.items()
            for index, node in enumerate(
                self.network.controllers[slot]["support_switches"]
            )
            if not update["action"] & (1 << index)
        }
        active_switches = self.connected - disabled
        self.active = self.configured.subgraph(
            active_switches | self.network.hosts.keys(),
        ).copy()
        for source, target, edge in list(self.active.edges(data=True)):
            endpoints = ((source, target), (target, source))
            for node, neighbor in endpoints:
                if node not in active_switches:
                    continue
                slot = self.network.switches[node]["controller"]
                active_ports = frames[self.owners[slot]]["ports"][node]
                if edge["ports"][node] not in active_ports:
                    self.active.remove_edge(node, neighbor)
                    break
        self.active.remove_nodes_from(
            host for host in self.network.hosts if not self.active.degree(host)
        )

    def observation_graph(self, slot: str) -> nx.Graph:
        """Include the subnet and links incident on its switches."""
        graph = self.active
        local = {
            node
            for node, row in self.network.switches.items()
            if row["controller"] == slot
        }
        local &= graph.nodes.keys()
        observed = nx.Graph()
        observed.add_nodes_from((node, graph.nodes[node]) for node in sorted(local))
        for source, target, edge in graph.edges(data=True):
            if source in local or target in local:
                observed.add_node(source, **graph.nodes[source])
                observed.add_node(target, **graph.nodes[target])
                observed.add_edge(source, target, **edge)
        return observed

    def resource_usage(self, slot: str) -> Tuple[int, int]:
        """Count active local switches and ports for modeled power."""
        switches = [
            node
            for node in self.active
            if node in self.network.switches
            and self.network.switches[node]["controller"] == slot
        ]
        return len(switches), sum(self.active.degree(node) for node in switches)
