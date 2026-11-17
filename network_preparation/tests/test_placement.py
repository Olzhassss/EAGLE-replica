"""Test unit-load capacity and deterministic hop-distance placement."""

import unittest

import networkx as nx

from network_preparation.balanced_louvain import balanced_partitions
from network_preparation.placement import place_controllers


class PlacementTests(unittest.TestCase):
    def test_capacity_holds_across_contraction_levels(self) -> None:
        graph = nx.path_graph(["s1", "s2", "s3", "s4", "s5", "s6"])
        levels = balanced_partitions(
            graph, capacity=2, difference=2, rho=0.01, resolution=1.0, seed=42
        )
        for level in levels:
            self.assertEqual(set().union(*level), set(graph))
            self.assertEqual(sum(map(len, level)), len(graph))
            self.assertLessEqual(max(map(len, level)), 2)

    def test_middle_switch_minimizes_mean_hops(self) -> None:
        graph = nx.path_graph(["s1", "s2", "s3"])
        placement = place_controllers(graph, [set(graph)])
        self.assertEqual(placement[0]["placement_switch"], "s2")
        self.assertAlmostEqual(placement[0]["mean_hops"], 2 / 3)
