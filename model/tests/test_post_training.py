"""Checkpoint-based policy continuation without a live network."""

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model.artifacts import read_rewards
from model.checkpoints import checkpoint_digest, load_checkpoint, save_checkpoint
from model.datasets import Dataset
from model.observations import ObservationEncoder
from model.scenario import ROOT, load_scenario
from model.settings import load_settings, training_stages
from model.train import train
from model.workers import AgentPool


class PostTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.network = load_scenario(ROOT / "network_preparation/source_v0.json")
        self.settings = load_settings()
        self.encoder = ObservationEncoder(self.network.controller_ids, 1, 1)
        self.transition = {
            "state": [0.1] * self.encoder.size,
            "next_state": [0.2] * self.encoder.size,
            "action": [1] * len(self.network.controller_ids),
            "reward": [0.5] * len(self.network.controller_ids),
            "terminated": False,
        }
        batch = {key: [value] for key, value in self.transition.items()}
        with AgentPool(
            self.encoder, self.settings["learning"], 42, distributed=False
        ) as agents:
            agents.run_round(1, "pretrain", batch)
            agents.run_round(2, "frozen", batch)
            state = copy.deepcopy(agents.state_dict())
        self.source_path = self.root / "checkpoints/source.pth"
        save_checkpoint(
            self.source_path,
            {
                "scenario_sha256": self.network.fingerprint,
                "settings": self.settings,
                "stages": training_stages("synthetic", 1, 1),
                "encoder": self.encoder.to_dict(),
                "agents": state,
                "round": 2,
                "completed_episodes": 2,
                "episode_rewards": [99.0],
            },
        )
        self.source_digest = checkpoint_digest(self.source_path)
        self.args = argparse.Namespace(
            network="v0.json",
            settings=ROOT / "settings/default.json",
            traffic=None,
            dataset="dc_t5",
            split="test",
            from_checkpoint="source",
            output="adapted",
            episodes=2,
            pretrain_episodes=None,
            resume=False,
            seed=42,
            workers="local",
            device="cpu",
        )
        self.dataset = Dataset(Path("dc_t5.json"), 100, [], "t5-hash")
        self.start_patch("model.train.MODELS", self.root)
        self.start_patch("model.train.load_network", return_value=self.network)
        self.load_dataset = self.start_patch(
            "model.train.load_dataset", return_value=self.dataset
        )
        self.start_patch("model.train.make_environment")
        self.start_patch(
            "model.train.make_encoder",
            side_effect=AssertionError("Unexpected recalibration"),
        )

    def start_patch(self, *args, **kwargs):
        patcher = patch(*args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def trial(self, *args, **kwargs):
        kwargs["on_transition"](self.transition)
        return {"cumulative_reward": float(kwargs["episode"])}

    def test_restores_full_state_and_keeps_estimator_frozen(self) -> None:
        source = load_checkpoint(self.source_path)
        with patch("model.train.run_trial", side_effect=self.trial) as trial:
            train(self.args)
        self.load_dataset.assert_called_once_with(self.network, "dc_t5", "test")
        self.assertEqual(trial.call_count, 2)
        result = load_checkpoint(self.root / "checkpoints/adapted.pth")
        self.assertEqual(result["round"], 4)
        self.assertEqual(result["encoder"], source["encoder"])
        self.assertEqual(result["episode_rewards"], [1.0, 2.0])
        self.assertEqual(
            result["source_checkpoint"],
            {"name": "source", "sha256": self.source_digest},
        )
        self.assertEqual(checkpoint_digest(self.source_path), self.source_digest)
        for slot, before in source["agents"].items():
            after = result["agents"][slot]
            for key, value in before["estimator"].items():
                torch.testing.assert_close(
                    after["estimator"][key], value, rtol=0, atol=0
                )
            torch.testing.assert_close(after["action_rng"], before["action_rng"])
            self.assertTrue(
                any(
                    not torch.equal(value, after["actor"][key])
                    for key, value in before["actor"].items()
                )
            )
            for name in ("actor", "critic", "estimator"):
                previous = before["optimizers"][name]["state"]
                current = after["optimizers"][name]["state"]
                for key in previous:
                    extra = 0 if name == "estimator" else 2
                    self.assertEqual(
                        current[key]["step"].item(),
                        previous[key]["step"].item() + extra,
                    )
        self.assertTrue((self.root / "adapted").exists())
        self.assertTrue((self.root / "adapted.png").exists())

    def test_interrupted_post_training_resumes_its_own_progress(self) -> None:
        def interrupted(*args, **kwargs):
            if kwargs["episode"] == 2:
                raise RuntimeError("interrupted")
            return self.trial(*args, **kwargs)

        with patch("model.train.run_trial", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                train(self.args)
        self.args.resume = True
        with patch("model.train.run_trial", side_effect=self.trial) as trial:
            train(self.args)
        self.assertEqual(trial.call_count, 1)
        self.assertEqual(trial.call_args.kwargs["episode"], 2)
        self.assertEqual(
            load_checkpoint(self.root / "checkpoints/adapted.pth")["round"], 4
        )
        _, rewards = read_rewards(self.root / "adapted.csv")
        self.assertEqual(rewards.tolist(), [1.0, 2.0])

    def test_rejects_incompatible_or_unready_source(self) -> None:
        original = load_checkpoint(self.source_path)
        for change, message in (
            ({"scenario_sha256": "different"}, "network"),
            ({"settings": {**self.settings, "embedding": {}}}, "embedding"),
            ({"completed_episodes": 0}, "pretraining"),
        ):
            with self.subTest(change=change):
                save_checkpoint(self.source_path, {**original, **change})
                with self.assertRaisesRegex(ValueError, message):
                    train(self.args)
        self.assertFalse((self.root / "checkpoints/adapted.pth").exists())

    def test_rejects_overwriting_source(self) -> None:
        self.args.output = "source"
        with self.assertRaisesRegex(ValueError, "new output name"):
            train(self.args)
        self.assertEqual(checkpoint_digest(self.source_path), self.source_digest)


if __name__ == "__main__":
    unittest.main()
