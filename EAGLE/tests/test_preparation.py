"""Exercise graph normalization, network generation and safe switch selection."""

import json
import tempfile
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from EAGLE.src import config
from EAGLE.src.data_preparation.hosts import hosts_for
from EAGLE.src.network_preparation.graphs import load_switch_graph, random_switch_graph
from EAGLE.src.network_preparation.placement import assign_hosts, select_controllable
from EAGLE.src.network_preparation.prepare import build_network, source_network
from EAGLE.src.paths import SOURCE_NETWORK


class PreparationTests(unittest.TestCase):
    def test_documented_random_variants_have_enough_safe_host_slots(self) -> None:
        for links in (16, 18):
            network = build_network(
                "fixture", random_switch_graph(13, links, 42), 9, "random"
            )
            self.assertEqual(len(network["hosts"]), 9)
            self.assertEqual(len(network["controllers"]), 3)

    def test_source_topologies_preserve_source_edges_and_use_dataset_addresses(
        self,
    ) -> None:
        hosts = [host.to_dict() for host in hosts_for(9)]
        source = json.loads(SOURCE_NETWORK.read_text())
        edges = lambda data: {
            frozenset((link["source"], link["target"]))
            for link in data["links"]
            if not link["target"].startswith("h")
        }
        v0 = source_network("v0", hosts)
        v3 = source_network("v3", hosts)
        self.assertEqual(edges(v0), edges(source))
        self.assertEqual((len(v0["switches"]), len(edges(v0))), (13, 16))
        self.assertEqual((len(v3["switches"]), len(edges(v3))), (16, 22))
        self.assertTrue(edges(v0) <= edges(v3))
        self.assertEqual(v0["hosts"], hosts)
        self.assertEqual(v3, source_network("v3", hosts))
        self.assertEqual(
            [row["support_switches"] for row in v3["controllers"]],
            [["s12"], ["s11"], ["s13"]],
        )

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
        self.assertEqual(
            nx.get_node_attributes(switch_graph, "source_node"),
            {1: "10", 2: "2", 3: "3"},
        )

    def test_switch_selection_backtracks_and_keeps_all_subsets_connected(self) -> None:
        graph = nx.cycle_graph(range(1, 6))
        domains = [{3, 4}, {1, 2, 5}]
        selected = select_controllable(graph, domains, 1)
        self.assertEqual(selected, [[3], [2]])
        for size in range(1, 3):
            for disabled in combinations((2, 3), size):
                self.assertTrue(
                    nx.is_connected(graph.subgraph(graph.nodes - set(disabled)))
                )

        graph = nx.path_graph(range(1, 6))
        with self.assertRaisesRegex(ValueError, "Cannot select support-switch counts"):
            select_controllable(graph, domains, 1)

    def test_host_slots_balance_across_available_peripheral_switches(self) -> None:
        graph = nx.path_graph(range(1, 7))
        domains = [{1, 2, 3}, {4, 5, 6}]
        self.assertEqual(assign_hosts(graph, domains, [[2], [6]], 3), [1, 4, 3])
        with self.assertRaisesRegex(ValueError, "only 4 access switches"):
            assign_hosts(graph, domains, [[2], [6]], 5)

    @patch.multiple(
        config,
        SEED=42,
        N_DOMAINS=3,
        LOUVAIN_CAPACITY=None,
        LOUVAIN_DIFFERENCE=10,
        LOUVAIN_RHO=0.9,
        LOUVAIN_RESOLUTION=0.1,
    )
    def test_random_network_is_reproducible_and_has_exact_counts(self) -> None:
        graph = random_switch_graph(13, 24, config.SEED)
        again = random_switch_graph(13, 24, config.SEED)
        self.assertEqual(set(graph.edges), set(again.edges))
        self.assertTrue(nx.is_connected(graph))
        network = build_network("example", graph, 9, "random")
        self.assertEqual(network, build_network("example", again, 9, "random"))
        self.assertEqual(len(network["switches"]), 13)
        self.assertEqual(len(network["links"]), 24 + 9)
        self.assertEqual(len(network["controllers"]), 3)
        owners = {row["id"]: row["controller"] for row in network["switches"]}
        self.assertEqual(
            Counter(
                owners[link["source"]]
                for link in network["links"]
                if link["target"].startswith("h")
            ),
            {"c0": 3, "c1": 3, "c2": 3},
        )
        self.assertTrue(
            all(
                58_000_000 <= link["capacity_bps"] <= 100_000_000
                for link in network["links"]
            )
        )

    @patch.multiple(
        config,
        SEED=42,
        N_DOMAINS=2,
        SWITCHES_PER_DOMAIN=2,
        LOUVAIN_CAPACITY=None,
        LOUVAIN_DIFFERENCE=10,
        LOUVAIN_RHO=0.9,
        LOUVAIN_RESOLUTION=0.1,
    )
    def test_support_switches_do_not_receive_host_attachments(self) -> None:
        graph = nx.convert_node_labels_to_integers(nx.barbell_graph(5, 0), first_label=1)
        network = build_network("barbell", graph, 6, "fixture")
        self.assertEqual(len(network["controllers"]), 2)
        for controller in network["controllers"]:
            support = controller["support_switches"]
            self.assertEqual(len(support), 2)
            self.assertEqual(controller["initial_action"], 3)
            self.assertTrue(
                all(
                    row["source"] not in support
                    for row in network["links"]
                    if row["target"].startswith("h")
                )
            )


if __name__ == "__main__":
    unittest.main()
