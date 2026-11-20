"""Regression checks for prepared inputs, resumption, and offline plot aggregation."""

import argparse
import copy
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from EAGLE.src.arguments import artifact_name
from EAGLE.src.artifacts import read_rewards
from EAGLE.src.datasets import DATASET_GROUPS, Dataset, load_dataset
from EAGLE.src.evaluation import topologies
from EAGLE.src.evaluation.plots import compare_condition
from EAGLE.src.evaluation.policy_comparison import evaluate_condition
from EAGLE.src.evaluation.topology_embeddings import plot_collected
from EAGLE.src.experiments.environment import JointEnvironment
from EAGLE.src.experiments.trials import run_trial
from EAGLE.src.measurement.flows import collect_flow_records
from EAGLE.src.model.checkpoints import load_checkpoint
from EAGLE.src.model.learner import AgentLearner
from EAGLE.src.model.observations import ObservationEncoder
from EAGLE.src.network import load_network
from EAGLE.src.paths import ROOT, SOURCE_NETWORK
from EAGLE.src.settings import load_settings
from EAGLE.src.training.stages import training_stages
from EAGLE.src.training.train import train


SOURCE = SOURCE_NETWORK


class PipelineTests(unittest.TestCase):
    def test_named_training_lock_rejects_a_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoints").mkdir()
            with (root / "checkpoints/example.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with (
                    patch("EAGLE.src.training.train.MODELS", root),
                    patch("EAGLE.src.training.train._train") as run,
                ):
                    with self.assertRaisesRegex(ValueError, "already running"):
                        train(argparse.Namespace(output="example"))
                    run.assert_not_called()

    def test_replay_without_tcp_payload_has_no_fct_samples(self) -> None:
        traffic = {
            "flows": [
                {"kind": "pcap", "resolved_path": "fixture.pcap", "sha256": "digest"}
            ]
        }
        with patch("EAGLE.src.measurement.flows._expected_payload", return_value={}):
            self.assertEqual(list(collect_flow_records(traffic, {})), [])

    def test_curriculum_budget_and_pretraining(self) -> None:
        stages = training_stages("curriculum", 12, 5)
        self.assertEqual(stages[0]["profiles"], ["dc_t0", *DATASET_GROUPS["congestions"]])
        self.assertEqual([stage["episodes"] for stage in stages[1:]], [3, 3, 2, 2, 2])
        self.assertTrue(all(stage["mode"] == "frozen" for stage in stages[1:]))
        with self.assertRaises(ValueError):
            training_stages("curriculum", 4, 1)

    def test_named_paths_and_pruning_settings(self) -> None:
        self.assertEqual(artifact_name("dc_t0"), "dc_t0")
        with self.assertRaises(argparse.ArgumentTypeError):
            artifact_name("../model")
        self.assertEqual(
            load_settings(ROOT / "settings/no_pruning.json")["learning"]["pruning"][
                "mode"
            ],
            "none",
        )

    def test_host_count_resolution_and_duration(self) -> None:
        network = load_network(SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "datasets/9_hosts/test/dc_t0.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "hosts": network.data["hosts"],
                        "duration_seconds": 100,
                        "jobs": [],
                    }
                )
            )
            dataset = load_dataset(network, "dc_t0", "test", root=root)
            self.assertEqual(dataset.decisions(load_settings()), 20)
            data = json.loads(path.read_text())
            data["hosts"][0]["mac"] = "02:00:00:00:00:01"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "addresses"):
                load_dataset(network, "dc_t0", "test", root=root)

    def test_pcap_replay_preserves_initial_idle_time(self) -> None:
        network = load_network(SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "datasets/9_hosts/test/dc_pcap.json"
            path.parent.mkdir(parents=True)
            (path.parent / "packets.pcap").write_bytes(b"fixture")
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "hosts": network.data["hosts"],
                        "duration_seconds": 100,
                        "captures": [
                            {
                                "source": "h1",
                                "target": "h2",
                                "path": "packets.pcap",
                                "packets": 1,
                                "first_packet_seconds": 2.5,
                            }
                        ],
                    }
                )
            )
            dataset = load_dataset(network, "dc_pcap", "test", root=root)
            (path.parent / "packets.pcap").write_bytes(b"changed fixture")
            changed = load_dataset(network, "dc_pcap", "test", root=root)
            self.assertNotEqual(dataset.fingerprint, changed.fingerprint)
        self.assertEqual(dataset.jobs[0]["start_seconds"], 2.5)

    def test_invalid_prepared_data_is_rejected_before_a_trial(self) -> None:
        network = load_network(SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "datasets/9_hosts/test/dc_t0.json"
            path.parent.mkdir(parents=True)
            valid = {
                "schema_version": 1,
                "hosts": network.data["hosts"],
                "duration_seconds": 100,
                "jobs": [],
            }
            invalid = [
                {"schema_version": 0},
                {"hosts": None},
                {"hosts": [network.data["hosts"][0]] * 9},
                {"duration_seconds": float("inf")},
                {
                    "jobs": [
                        {
                            "id": "bad-rate",
                            "kind": "tcp",
                            "source": "h1",
                            "target": "h2",
                            "start_seconds": 0,
                            "size_bytes": 10,
                            "rate_bps": -1,
                            "timeout_seconds": 60,
                        }
                    ],
                },
            ]
            for updates in invalid:
                with self.subTest(updates=updates):
                    path.write_text(json.dumps(dict(valid, **updates)))
                    with self.assertRaises(ValueError):
                        load_dataset(network, "dc_t0", "test", root=root)

    def test_settings_require_objects_for_nested_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            for data in ([], {"learning": None}, {"embedding": 1}):
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "JSON object"):
                        load_settings(path)

    def test_trial_rejects_fractional_actions_without_rounding(self) -> None:
        network = load_network(SOURCE)
        encoder = ObservationEncoder(network.controller_ids, 1, 1)
        client, commander = MagicMock(), MagicMock()
        environment = JointEnvironment(
            network, load_settings(), client, commander, encoder
        )
        with patch.object(
            environment, "reset", return_value=({"global_obs": np.zeros(12)}, {})
        ):
            with self.assertRaisesRegex(ValueError, "integer support-switch mask"):
                run_trial(
                    environment,
                    lambda _: [0.5, 1.0, 0.0],
                    MagicMock(),
                    episode=1,
                    dataset=Dataset(Path("dc_t0.json"), 100, [], "digest"),
                )
        client.apply_actions.assert_not_called()
        commander.finish_traffic.assert_called_once_with(0)

    def test_comparison_reserves_the_baseline_name(self) -> None:
        args = argparse.Namespace(
            models=["baseline", "b", "c"],
            episodes=1,
            repetitions=1,
            seed=42,
            device="cpu",
        )
        with patch("EAGLE.src.evaluation.policy_comparison.load_inference_model") as load:
            with self.assertRaisesRegex(ValueError, "baseline is reserved"):
                evaluate_condition(
                    args,
                    load_network(SOURCE),
                    load_settings(),
                    Dataset(Path("dc_t0.json"), 100, [], "digest"),
                    "suite",
                )
        load.assert_not_called()

    def test_topology_rewards_keep_custom_model_names(self) -> None:
        network = load_network(SOURCE)
        models = ["first", "second", "third"]
        condition = {
            "measurements": {model: [{"episode_rewards": [1, 2]}] for model in models}
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (
                patch(
                    "sys.argv",
                    ["eval-topologies", "--network", "v0.json", "--models", *models],
                ),
                patch.object(topologies, "EVAL", output),
                patch.object(topologies, "load_network", return_value=network),
                patch.object(topologies, "load_dataset"),
                patch.object(topologies, "evaluate_condition"),
                patch.object(topologies, "compare_condition", return_value=condition),
            ):
                topologies.main()
            self.assertEqual(
                {path.stem for path in output.glob("*.csv")},
                {f"{model}_{network.path.stem}" for model in models},
            )

    def test_frozen_estimator_stays_fixed_while_policy_learns(self) -> None:
        learner = AgentLearner(
            "c0",
            ObservationEncoder(("c0",), 1, 1),
            load_settings()["learning"],
            42,
            support_counts=(1,),
        )
        estimator = copy.deepcopy(learner.estimator.state_dict())
        batch = {
            "state": [[0.1] * 4],
            "next_state": [[0.2] * 4],
            "action": [[1]],
            "reward": [[0.5]],
            "terminated": [False],
        }
        learner.learn("frozen", batch)
        for name, tensor in estimator.items():
            torch.testing.assert_close(tensor, learner.estimator.state_dict()[name])
        self.assertFalse(learner.optimizers["estimator"].state)

    def test_trial_drains_and_ignores_incomplete_deliveries(self) -> None:
        environment = MagicMock()
        environment.settings = load_settings()
        environment.reset.return_value = ({"global_obs": np.zeros(4)}, {})
        environment.step.return_value = (
            {"global_obs": np.ones(4)},
            np.array([1.0, 3.0]),
            False,
            False,
            {
                "step_seconds": 5,
                "raw_embedding": [1.0] * 4,
                "reward_components": [{"power_watts": 20, "total_delivered_bps": 1}],
            },
        )
        dataset = Dataset(Path("dc_t0.json"), 100, [], "digest")
        complete = {"fct_seconds": 2.0}
        with patch(
            "EAGLE.src.experiments.trials.collect_flow_records",
            return_value=[complete, {"fct_seconds": None}],
        ):
            result = run_trial(
                environment,
                lambda _: [1, 1],
                MagicMock(),
                episode=1,
                dataset=dataset,
                observations=16,
            )
        self.assertEqual(len(result["embeddings"]), 16)
        self.assertEqual(result["cumulative_reward"], 32)
        self.assertEqual(result["fct_seconds"], 2)
        environment.commander.finish_traffic.assert_called_once_with(5)

    def test_paired_efficiency_normalization(self) -> None:
        def trial(fct, power):
            return [{"seed": 42, "fct_seconds": fct, "power_watts": power}]

        condition = {
            "measurements": {
                "baseline": trial(10, 100),
                "a": trial(5, 50),
                "b": trial(8, 80),
                "c": trial(9, 100),
            }
        }
        result = compare_condition(condition, ["a", "b", "c"], 42)
        self.assertEqual(result["largest_mean_fct_gain"], 0.5)
        self.assertEqual(result["metrics"]["a"]["efficiency"]["mean"], 2)
        self.assertEqual([len(group) for group in DATASET_GROUPS.values()], [2, 3, 4])

    def test_repetition_seed_is_shared_by_models_and_baseline(self) -> None:
        network = load_network(SOURCE)
        encoder = ObservationEncoder(network.controller_ids, 1, 1)
        args = argparse.Namespace(
            models=["a", "b", "c"], episodes=1, repetitions=2, seed=123, device="cpu"
        )
        policy = MagicMock()
        episode = {
            "fct_seconds": 2,
            "power_watts": 50,
            "completed_flows": 4,
            "measured_seconds": 100,
            "cumulative_reward": 3,
        }
        with (
            patch(
                "EAGLE.src.evaluation.policy_comparison.load_inference_model",
                return_value=(policy, encoder),
            ),
            patch(
                "EAGLE.src.evaluation.policy_comparison.checkpoint_digest",
                return_value="digest",
            ),
            patch("EAGLE.src.evaluation.policy_comparison.seed_everything") as seed,
            patch("EAGLE.src.evaluation.policy_comparison.make_environment"),
            patch("EAGLE.src.evaluation.policy_comparison.RunLog"),
            patch(
                "EAGLE.src.evaluation.policy_comparison.run_trial", return_value=episode
            ),
        ):
            result = evaluate_condition(
                args,
                network,
                load_settings(),
                Dataset(Path("dc_t0.json"), 100, [], "data"),
                "suite",
            )
        self.assertEqual(
            [call.args[0] for call in seed.call_args_list], [123] * 4 + [124] * 4
        )
        self.assertEqual(
            [row["seed"] for row in result["measurements"]["baseline"]], [123, 124]
        )
        self.assertEqual(result["measurements"]["a"][0]["fct_seconds"], 2)

    def test_topology_embedding_collection_projects_all_eighty_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(name="embeddings", model="dc_t0", seed=42)
            for index in range(5):
                data = {
                    "network": f"v{index}",
                    "model": "dc_t0",
                    "model_sha256": "digest",
                    "encoder": {},
                    "seed": 42,
                    "embeddings": [[index] * 12] * 16,
                }
                (root / f"embeddings_v{index}.json").write_text(json.dumps(data))
            with (
                patch("EAGLE.src.evaluation.topology_embeddings.EVAL", root),
                patch(
                    "EAGLE.src.evaluation.topology_embeddings.plot_embeddings"
                ) as project,
            ):
                plot_collected(args)
            samples = project.call_args.args[0]
            self.assertEqual(len(samples), 80)
            self.assertEqual(
                {row["label"] for row in samples}, {f"v{i}" for i in range(5)}
            )

    def test_checkpoint_resume_preserves_reward_history(self) -> None:
        network = load_network(SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "settings.json"
            settings.write_text(json.dumps({"embedding": {"normalization": "identity"}}))
            args = argparse.Namespace(
                network="v0.json",
                traffic="curriculum",
                output="example",
                settings=settings,
                episodes=5,
                pretrain_episodes=1,
                seed=42,
                resume=False,
                device="cpu",
                workers="local",
                mininet_socket=None,
                ryu_socket=None,
            )
            pool = MagicMock()
            pool.__enter__.return_value = pool
            pool.state_dict.return_value = {
                slot: {"actor": {}, "critic": {}} for slot in network.controller_ids
            }
            pool.run_round.return_value = {}

            def trial(*unused, **kwargs):
                if kwargs["episode"] == 4 and not args.resume:
                    raise RuntimeError("interrupted")
                return {"cumulative_reward": float(kwargs["episode"]), "embeddings": []}

            with (
                patch("EAGLE.src.training.train.MODELS", root / "models"),
                patch("EAGLE.src.training.train.load_network", return_value=network),
                patch(
                    "EAGLE.src.training.train.load_dataset",
                    side_effect=lambda _, name, split: Dataset(
                        Path(name + ".json"), 100, [], name
                    ),
                ),
                patch("EAGLE.src.training.train.make_environment"),
                patch("EAGLE.src.training.train.AgentPool", return_value=pool),
                patch("EAGLE.src.training.train.run_trial", side_effect=trial),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    train(args)
                checkpoint = load_checkpoint(root / "models/checkpoints/example.pth")
                self.assertEqual(checkpoint["completed_episodes"], 3)
                self.assertEqual(checkpoint["episode_rewards"], [2, 3])
                args.resume = True
                train(args)
            _, rewards = read_rewards(root / "models/example.csv")
            np.testing.assert_array_equal(rewards, [2, 3, 4, 5, 6])
            self.assertTrue((root / "models/example.png").exists())
            self.assertTrue((root / "models/example").exists())


if __name__ == "__main__":
    unittest.main()
