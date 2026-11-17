"""Exercise graph normalization, scenario generation and safe switch selection."""

import tempfile
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from config import Config
from network_preparation.placement import assign_hosts, select_controllable
from network_preparation.prepare import build_scenario, load_switch_graph, random_switch_graph


class PreparationTests(unittest.TestCase):
    def test_graphml_normalization(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edges_from([("10", "2"), ("2", "10"), ("2", "3"), ("2", "2")])
        graph.add_node("host", isHost=True)
        graph.add_edge("3", "host")
        graph.add_node("isolated")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.graphml"
            nx.write_graphml(graph, path)
            switch_graph = load_switch_graph(path)
        self.assertEqual(list(switch_graph.edges), [(1, 2), (2, 3)])
        self.assertEqual(nx.get_node_attributes(switch_graph, "source_node"), {1: "10", 2: "2", 3: "3"})

    def test_switch_selection_backtracks_and_keeps_all_subsets_connected(self) -> None:
        graph = nx.cycle_graph(range(1, 6))
        domains = [{3, 4}, {1, 2, 5}]
        selected = select_controllable(graph, domains, 1)
        self.assertEqual(selected, [[3], [2]])
        for size in range(1, 3):
            for disabled in combinations((2, 3), size):
                self.assertTrue(nx.is_connected(graph.subgraph(graph.nodes - set(disabled))))

        graph = nx.path_graph(range(1, 6))
        with self.assertRaisesRegex(ValueError, "Cannot select 1"):
            select_controllable(graph, domains, 1)

    def test_host_slots_balance_across_available_peripheral_switches(self) -> None:
        graph = nx.path_graph(range(1, 7))
        domains = [{1, 2, 3}, {4, 5, 6}]
        self.assertEqual(assign_hosts(graph, domains, [[2], [6]], 3), [1, 4, 3])
        with self.assertRaisesRegex(ValueError, "only 4 access switches"):
            assign_hosts(graph, domains, [[2], [6]], 5)

    @patch.multiple(Config, SEED=42, N_DOMAINS=3, SWITCHES_PER_DOMAIN=1,
                    LOUVAIN_CAPACITY=None, LOUVAIN_DIFFERENCE=10,
                    LOUVAIN_RHO=0.9, LOUVAIN_RESOLUTION=0.1)
    def test_random_scenario_is_reproducible_and_has_exact_counts(self) -> None:
        graph = random_switch_graph(13, 24, Config.SEED)
        again = random_switch_graph(13, 24, Config.SEED)
        self.assertEqual(set(graph.edges), set(again.edges))
        self.assertTrue(nx.is_connected(graph))
        scenario = build_scenario("example", graph, 9, "random")
        self.assertEqual(scenario, build_scenario("example", again, 9, "random"))
        self.assertEqual(len(scenario["switches"]), 13)
        self.assertEqual(len(scenario["links"]), 24 + 9)
        self.assertEqual(len(scenario["controllers"]), 3)
        owners = {row["id"]: row["controller"] for row in scenario["switches"]}
        self.assertEqual(
            Counter(owners[link["source"]] for link in scenario["links"] if link["target"].startswith("h")),
            {"c0": 3, "c1": 3, "c2": 3},
        )
        self.assertTrue(all(58_000_000 <= link["capacity_bps"] <= 100_000_000 for link in scenario["links"]))

    @patch.multiple(Config, SEED=42, N_DOMAINS=2, SWITCHES_PER_DOMAIN=2,
                    LOUVAIN_CAPACITY=None, LOUVAIN_DIFFERENCE=10,
                    LOUVAIN_RHO=0.9, LOUVAIN_RESOLUTION=0.1)
    def test_config_supports_multiple_controllable_switches_per_domain(self) -> None:
        graph = nx.convert_node_labels_to_integers(nx.barbell_graph(5, 0), first_label=1)
        scenario = build_scenario("barbell", graph, 6, "fixture")
        self.assertEqual(len(scenario["controllers"]), 2)
        for controller in scenario["controllers"]:
            switches = controller["controllable_switches"]
            self.assertEqual(len(switches), 2)
            self.assertEqual(switches[0], controller["support_switch"])
            self.assertTrue(all(
                row["source"] not in switches
                for row in scenario["links"] if row["target"].startswith("h")
            ))


if __name__ == "__main__":
    unittest.main()
