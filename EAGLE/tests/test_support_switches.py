"""Configurable support layouts, categorical policies, and worker/checkpoint contracts."""

import copy
import tempfile
import unittest
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import torch

from EAGLE.src.communication.controller_client import ControllerClient
from EAGLE.src.datasets import Dataset
from EAGLE.src.experiments.environment import JointEnvironment
from EAGLE.src.model.actor_critic import ActorCritic
from EAGLE.src.model.checkpoints import load_inference_model, save_checkpoint
from EAGLE.src.model.learner import AgentLearner
from EAGLE.src.model.observations import ObservationEncoder
from EAGLE.src.network import Network, load_network
from EAGLE.src.network_preparation.placement import select_controllable
from EAGLE.src.network_preparation.prepare import source_network
from EAGLE.src.paths import SOURCE_NETWORK
from EAGLE.src.settings import load_settings
from EAGLE.src.training.agent_pool import AgentPool
from network_schema import resolve_support_counts, validate_support_layout


def multi_network() -> Network:
    source = load_network(SOURCE_NETWORK)
    data = copy.deepcopy(source.data)
    for row, nodes in zip(
        data["controllers"], (["s12", "s10"], ["s11"], []), strict=True
    ):
        row["support_switches"] = nodes
        row["initial_action"] = (1 << len(nodes)) - 1
    validate_support_layout(data)
    return replace(source, data=data, fingerprint="multi-support-fixture")


class SupportPreparationTests(unittest.TestCase):
    def test_counts_can_be_uniform_mixed_or_zero(self):
        self.assertEqual(resolve_support_counts(2, 3), (2, 2, 2))
        self.assertEqual(resolve_support_counts([2], 3), (2, 2, 2))
        self.assertEqual(resolve_support_counts([2, 1, 0], 3), (2, 1, 0))
        for invalid in (True, -1, 1.5, [1, 2], [1, -1, 2], [1, True, 2], 63):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_support_counts(invalid, 3)

    def test_multiple_supports_preserve_every_disable_combination(self):
        graph = nx.convert_node_labels_to_integers(nx.barbell_graph(5, 0), first_label=1)
        domains = [set(range(1, 6)), set(range(6, 11))]
        selected = select_controllable(graph, domains, (2, 2))
        self.assertEqual(list(map(len, selected)), [2, 2])
        supports = [node for group in selected for node in group]
        for size in range(len(supports) + 1):
            for disabled in combinations(supports, size):
                self.assertTrue(
                    nx.is_connected(graph.subgraph(graph.nodes - set(disabled)))
                )
        self.assertEqual(
            list(map(len, select_controllable(graph, domains, (0, 2)))), [0, 2]
        )

    def test_source_variants_respect_fixed_host_attachments(self):
        hosts = load_network(SOURCE_NETWORK).data["hosts"]
        extended = source_network("v3", hosts, support_switches=2)
        self.assertEqual(
            [len(row["support_switches"]) for row in extended["controllers"]], [2, 2, 2]
        )
        validate_support_layout(extended)
        with self.assertRaisesRegex(ValueError, "Cannot select"):
            source_network("v0", hosts, support_switches=2)

    def test_schema_rejects_bad_support_layouts(self):
        original = multi_network().data
        changes = [
            {"support_switches": ["s12", "s12"]},
            {"support_switches": ["s11"]},
            {"support_switches": ["s1"], "initial_action": 1},
            {"initial_action": 4},
            {"initial_action": True},
            {"support_switches": "s12"},
        ]
        for updates in changes:
            data = copy.deepcopy(original)
            data["controllers"][0].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_support_layout(data)
        with self.assertRaisesRegex(ValueError, "regenerate older networks"):
            validate_support_layout(dict(original, schema_version=1))


class SupportRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.network = multi_network()
        self.encoder = ObservationEncoder(self.network.controller_ids, 1, 1)

    def test_action_space_and_reset_enable_every_configured_support(self):
        commander = MagicMock()
        commander.start_traffic.return_value = {"started_at": 0}
        client = MagicMock()
        client.apply_actions.return_value = {
            slot: {"accepted_at": 0} for slot in self.network.controller_ids
        }
        environment = JointEnvironment(
            self.network, load_settings(), client, commander, self.encoder
        )
        np.testing.assert_array_equal(environment.action_space.nvec, [4, 2, 1])
        self.assertTrue(environment.action_space.contains(np.array([2, 1, 0])))
        self.assertFalse(environment.action_space.contains(np.array([4, 0, 0])))
        with (
            patch.object(environment, "_wait_for_samples", return_value=[]),
            patch.object(environment, "_observe", return_value=({}, [])),
        ):
            environment.reset(Dataset(Path("fixture.json"), 100, [], "fixture"))
        self.assertEqual(
            client.apply_actions.call_args.args[0], {"c0": 3, "c1": 1, "c2": 0}
        )
        for actions in ([4, 0, 0], [0, 2, 0], [0, 0, 1], [1.5, 0, 0]):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                environment.step(actions)
        self.assertEqual(client.apply_actions.call_count, 1)

    def test_client_rejects_invalid_masks_before_network_io(self):
        client = ControllerClient(self.network)
        with patch.object(client, "_attempt") as request:
            for actions in ({"c0": 4}, {"c1": 2}, {"c2": 1}, {"c0": True}, {"other": 0}):
                with self.subTest(actions=actions), self.assertRaises(ValueError):
                    client.apply_actions(actions, float("inf"))
        request.assert_not_called()

    def test_checkpoint_keeps_action_dimensions_and_rejects_changed_counts(self):
        policy = ActorCritic(self.encoder.size, self.network.action_sizes)
        for actor, action in zip(policy.actor, self.network.all_on_actions, strict=True):
            with torch.no_grad():
                actor[-1].weight.zero_()
                actor[-1].bias.zero_()
                actor[-1].bias[action] = 10
        observation = {"global_obs": np.ones(self.encoder.size, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pth"
            save_checkpoint(
                path,
                {
                    "encoder": self.encoder.to_dict(),
                    "support_counts": self.network.support_counts,
                    "agents": {
                        slot: {"actor": actor.state_dict(), "critic": critic.state_dict()}
                        for slot, actor, critic in zip(
                            self.network.controller_ids,
                            policy.actor,
                            policy.critic,
                            strict=True,
                        )
                    },
                },
            )
            restored, _ = load_inference_model(path, self.network)
            np.testing.assert_array_equal(restored.decide(observation), [3, 1, 0])
            changed = copy.deepcopy(self.network.data)
            changed["controllers"][0]["support_switches"] = ["s12"]
            changed["controllers"][0]["initial_action"] = 1
            with self.assertRaisesRegex(ValueError, "support-switch counts"):
                load_inference_model(path, replace(self.network, data=changed))

    def test_reward_estimator_sees_individual_switch_bits(self):
        learner = AgentLearner(
            "c0", self.encoder, load_settings()["learning"], 42, support_counts=(2, 1, 0)
        )
        seen = []
        handle = learner.estimator.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone())
        )
        batch = {
            "state": [[0.1] * self.encoder.size] * 2,
            "next_state": [[0.2] * self.encoder.size] * 2,
            "action": [[2, 1, 0], [3, 0, 0]],
            "reward": [[0.5] * 3] * 2,
            "terminated": [False, False],
        }
        learner.learn("pretrain", batch)
        handle.remove()
        torch.testing.assert_close(
            seen[0][:, self.encoder.size :],
            torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        )
        self.assertEqual(learner.actor[-1].out_features, 4)
        before = copy.deepcopy(learner.estimator.state_dict())
        learner.learn("frozen", batch)
        for name, tensor in before.items():
            torch.testing.assert_close(tensor, learner.estimator.state_dict()[name])

    def test_local_and_process_workers_learn_and_resume_mixed_counts(self):
        encoder = ObservationEncoder(("c0", "c1"), 1, 1)
        observation = {"global_obs": np.ones(encoder.size, dtype=np.float32)}
        batch = {
            "state": [[0.1] * encoder.size],
            "next_state": [[0.2] * encoder.size],
            "action": [[2, 0]],
            "reward": [[0.5, 0.2]],
            "terminated": [False],
        }
        states = []
        for distributed in (False, True):
            with AgentPool(
                encoder,
                load_settings()["learning"],
                42,
                support_counts=(2, 0),
                distributed=distributed,
            ) as agents:
                self.assertEqual(len(agents.decide(observation)), 2)
                agents.run_round(1, "pretrain", batch)
                agents.run_round(2, "frozen", batch)
                saved = copy.deepcopy(agents.state_dict())
                expected = agents.decide(observation)
                agents.load_state_dict(saved)
                self.assertEqual(agents.decide(observation), expected)
                self.assertEqual(saved["c0"]["support_counts"], (2, 0))
                states.append(saved)
        for slot in encoder.controller_ids:
            for component in ("actor", "critic", "estimator"):
                for name, tensor in states[0][slot][component].items():
                    torch.testing.assert_close(tensor, states[1][slot][component][name])


if __name__ == "__main__":
    unittest.main()
