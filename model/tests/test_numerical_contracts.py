"""Targeted checks for behavior that crosses the training and checkpoint boundary."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model.actor_critic import ActorCritic
from model.checkpoints import load_inference_model, save_checkpoint
from model.consensus import mix_contributions, pack_contribution, pruning_fraction
from model.observations import ObservationEncoder
from model.reward import compute_reward
from model.settings import DEFAULT_SCENARIO, load_scenario


class NumericalContracts(unittest.TestCase):
    def test_feature_count_tracks_controller_slots_and_order(self) -> None:
        encoder = ObservationEncoder(("c0", "c1", "c2"), 2, 1, False)
        graph = {
            "nodes": [{"id": "s2"}, {"id": "s1"}],
            "links": [{"source": "s1", "target": "s2", "weight": 0.5}],
        }
        embedded = encoder.embed_local(graph)
        self.assertEqual(embedded.shape, (8,))
        observed = encoder.normalize(np.concatenate([embedded] * 3))
        self.assertEqual(observed.shape, (24,))
        self.assertEqual(observed.dtype, np.float32)

    def test_reward_uses_acknowledged_support_and_recorded_poll_window(self) -> None:
        snapshot = {
            "sample_duration_seconds": 2,
            "controller_id": "c0",
            "delivered_bytes": {"h1 h2": 500},
            "throughput": {"h1 h2": 500},
            "usage": {},
            "resources": {"c0": {"action": 1}},
        }
        reward = compute_reward(
            snapshot,
            {"kind": "legacy", "throughput_scale": 40, "overprovisioning_penalty": 5},
        )
        self.assertEqual(reward["mean_delivered_flow_bps"], 2000)
        self.assertEqual(reward["reward"], -4)

    def test_sparsification_leaves_sender_state_and_first_layer_dense(self) -> None:
        sender = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        receiver = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        with torch.no_grad():
            sender[0].weight.fill_(3.0)
            sender[1].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        pack = pack_contribution(6, {"critic": sender}, 4.0, 0.5)
        mu = mix_contributions(6, {"critic": receiver}, 0.0, 0.0, [(1.0, pack)])
        self.assertEqual(mu, 4.0)
        torch.testing.assert_close(receiver[0].weight, sender[0].weight)
        torch.testing.assert_close(
            receiver[1].weight, torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        )
        torch.testing.assert_close(
            sender[1].weight, torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        rates = {"mode": "dynamic", "warmup_rounds": 5, "ramp_rounds": 10}
        self.assertEqual(pruning_fraction(rates, 5), 0)
        self.assertEqual(pruning_fraction(rates, 6), 0.3)
        self.assertAlmostEqual(pruning_fraction(rates, 16), 0.7)

    def test_combined_legacy_checkpoint_round_trip(self) -> None:
        scenario = load_scenario(DEFAULT_SCENARIO)
        encoder = ObservationEncoder(scenario.controller_ids, 1, 1, True)
        source = ActorCritic(encoder.size, len(encoder.controller_ids))
        obs = {"global_obs": np.ones(encoder.size, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pth"
            save_checkpoint(checkpoint, {"model": source.state_dict()})
            restored, _ = load_inference_model(checkpoint, {}, scenario, encoder)
        np.testing.assert_array_equal(restored.decide(obs), source.decide(obs))
