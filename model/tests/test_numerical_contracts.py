"""Targeted checks for behavior that crosses the training and checkpoint boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from model.actor_critic import ActorCritic
from model.checkpoints import load_inference_model, save_checkpoint
from model.consensus import mix_contributions, pack_contribution, pruning_fraction
from model.emulation import Emulation
from model.observations import ObservationEncoder, embed_subnet
from model.reward import compute_reward
from model.scenario import DEFAULT_SCENARIO, load_scenario


class NumericalContracts(unittest.TestCase):
    def test_feature_count_tracks_controller_slots_and_order(self) -> None:
        encoder = ObservationEncoder(("c0", "c1", "c2"), 2, 1)
        graph = {
            "nodes": [{"id": "s2"}, {"id": "s1"}],
            "links": [{"source": "s1", "target": "s2", "weight": 0.5}],
        }
        embedded = encoder.embed_local(graph)
        self.assertEqual(embedded.shape, (8,))
        observed = encoder.normalize(np.concatenate([embedded] * 3))
        self.assertEqual(observed.shape, (24,))
        self.assertEqual(observed.dtype, np.float32)

    def test_feather_uses_karate_club_degree_with_implicit_loops(self) -> None:
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "links": [{"source": "a", "target": "b", "weight": 0.5}],
        }
        angles = np.array([np.log(4.0) * 0.01, 0.0])
        expected = 0.5 * np.concatenate((np.cos(angles), np.sin(angles)))
        np.testing.assert_allclose(embed_subnet(graph, order=1, eval_points=1), expected)

    def test_reward_uses_delivered_rate_and_modeled_power(self) -> None:
        snapshot = {
            "sample_duration_seconds": 2,
            "delivered_bytes": {"h1 h2": 500},
            "usage": {
                "active_switches": 2,
                "active_ports": 4,
                "switch_egress_bps": {"s1": 1_000_000},
            },
        }
        reward = compute_reward(
            snapshot,
            {
                "alpha": 0.5,
                "throughput_reference_bps": 4000,
                "power_reference_watts": 80,
                "power_model": {
                    "idle_switch_watts": 10,
                    "idle_port_watts": 2,
                    "dynamic_watts_per_mbps": 12,
                },
            },
        )
        self.assertEqual(reward["mean_delivered_flow_bps"], 2000)
        self.assertEqual(reward["power_watts"], 40)
        self.assertEqual(reward["reward"], 0.25)

    def test_emulation_discovers_launcher_addresses_with_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mininet_nets").mkdir()
            (root / "mininet_nets/.socket").write_text("/tmp/mininet.sock\n")
            (root / ".socket").write_text("/tmp/ryu.sock")
            with patch("model.emulation.ROOT", root):
                emulation = Emulation()
                self.assertEqual(emulation.mininet_socket, Path("/tmp/mininet.sock"))
                self.assertEqual(emulation.ryu_socket, Path("/tmp/ryu.sock"))
                self.assertEqual(
                    Emulation(ryu_socket="/tmp/override.sock").ryu_socket,
                    Path("/tmp/override.sock"),
                )
                self.assertIsNone(Emulation(discover=False).mininet_socket)

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
            receiver[1].weight,
            torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            sender[1].weight,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        rates = {"mode": "dynamic", "warmup_rounds": 5, "ramp_rounds": 10}
        self.assertEqual(pruning_fraction(rates, 5), 0)
        self.assertEqual(pruning_fraction(rates, 6), 0.3)
        self.assertAlmostEqual(pruning_fraction(rates, 16), 0.7)

    def test_native_checkpoint_round_trip(self) -> None:
        scenario = load_scenario(DEFAULT_SCENARIO)
        encoder = ObservationEncoder(scenario.controller_ids, 1, 1)
        source = ActorCritic(encoder.size, len(encoder.controller_ids))
        obs = {"global_obs": np.ones(encoder.size, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                checkpoint,
                {
                    "encoder": encoder.to_dict(),
                    "agents": {
                        slot: {
                            "actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                        }
                        for slot, actor, critic in zip(
                            scenario.controller_ids,
                            source.actor,
                            source.critic,
                            strict=True,
                        )
                    },
                },
            )
            restored, saved_encoder = load_inference_model(checkpoint, scenario)
        self.assertEqual(saved_encoder.to_dict(), encoder.to_dict())
        np.testing.assert_array_equal(restored.decide(obs), source.decide(obs))

    def test_rejects_checkpoint_without_agent_state(self) -> None:
        scenario = load_scenario(DEFAULT_SCENARIO)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "incomplete.pth"
            save_checkpoint(checkpoint, {"encoder": {}})
            with self.assertRaisesRegex(ValueError, "encoder and agent state"):
                load_inference_model(checkpoint, scenario)
