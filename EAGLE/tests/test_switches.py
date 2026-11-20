"""Verify fixed-horizon switch-count comparisons with mocked runtime clients."""

import argparse
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from EAGLE.src.datasets import Dataset
from EAGLE.src.evaluation.switches import evaluate_switches, main
from EAGLE.src.experiments.environment import JointEnvironment
from EAGLE.src.experiments.trials import run_trial
from EAGLE.src.model.observations import ObservationEncoder
from EAGLE.src.network import load_network
from EAGLE.src.paths import DEFAULT_SETTINGS, ROOT, SOURCE_NETWORK
from EAGLE.src.settings import load_settings


class SwitchEvaluationTests(unittest.TestCase):
    def arguments(self, **updates) -> argparse.Namespace:
        values = dict(
            network="v3.json",
            steps=3,
            models=["first", "second", "third"],
            dataset="dc_pcap",
            name=None,
            seed=123,
            settings=DEFAULT_SETTINGS,
            device="cpu",
            mininet_socket=None,
            ryu_socket=None,
        )
        return argparse.Namespace(**dict(values, **updates))

    def network(self):
        source = load_network(SOURCE_NETWORK)
        return replace(source, path=ROOT / "networks/v3.json")

    def test_each_policy_runs_one_episode_and_counts_enabled_switches(self) -> None:
        network = self.network()
        encoder = ObservationEncoder(network.controller_ids, 1, 1)
        actions = [
            [[1, 0, 1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[1, 1, 1], [0, 0, 1], [1, 0, 0]],
        ]
        policies = [MagicMock() for _ in actions]
        for policy, decisions in zip(policies, actions, strict=True):
            policy.decide.side_effect = decisions
        dataset = Dataset(Path("dc_pcap.json"), 5, [], "traffic-hash")

        def trial(environment, decide, log, **kwargs):
            self.assertEqual(kwargs["episode"], 1)
            self.assertEqual(kwargs["steps"], 3)
            self.assertFalse(kwargs["measure_flows"])
            for _ in range(kwargs["steps"]):
                kwargs["on_transition"]({"action": decide({})})

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (
                patch("EAGLE.src.evaluation.switches.EVAL", output),
                patch("EAGLE.src.evaluation.switches.load_network", return_value=network),
                patch(
                    "EAGLE.src.evaluation.switches.load_dataset", return_value=dataset
                ) as load,
                patch(
                    "EAGLE.src.evaluation.policy_comparison.load_inference_model",
                    side_effect=[(policy, encoder) for policy in policies],
                ),
                patch(
                    "EAGLE.src.evaluation.switches.checkpoint_digest",
                    return_value="model-hash",
                ),
                patch("EAGLE.src.evaluation.switches.make_environment"),
                patch("EAGLE.src.evaluation.switches.RunLog"),
                patch("EAGLE.src.evaluation.switches.seed_everything") as seed,
                patch(
                    "EAGLE.src.evaluation.switches.run_trial", side_effect=trial
                ) as run,
            ):
                report = evaluate_switches(self.arguments())
            expected = {
                "baseline": [3, 3, 3],
                "first": [2, 1, 0],
                "second": [0, 2, 3],
                "third": [3, 1, 1],
            }
            self.assertEqual(report["enabled_switches"], expected)
            self.assertEqual(
                report["final_enabled_switches"],
                {
                    "baseline": 3,
                    "first": 0,
                    "second": 3,
                    "third": 1,
                },
            )
            self.assertEqual(
                json.loads((output / "switches_v3.json").read_text()), report
            )
            self.assertTrue((output / "switches_v3.png").is_file())
        load.assert_called_once_with(network, "dc_pcap", "test")
        self.assertEqual(seed.call_args_list, [call(123)] * 4)
        self.assertEqual(run.call_count, 4)
        for policy in policies:
            policy.to.assert_called_once_with(torch.device("cpu"))

    def test_invalid_counts_and_model_names_fail_before_runtime_access(self) -> None:
        invalid = [
            {"steps": 0},
            {"steps": -1},
            {"steps": 1.5},
            {"models": ["first", "first", "second"]},
            {"models": ["baseline", "first", "second"]},
            {"models": ["first", "second"]},
        ]
        with patch("EAGLE.src.evaluation.switches.make_environment") as environment:
            for updates in invalid:
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    evaluate_switches(self.arguments(**updates))
        environment.assert_not_called()

    def test_switch_counts_decode_masks_instead_of_adding_action_ids(self) -> None:
        network = self.network()
        data = copy.deepcopy(network.data)
        data["controllers"][0].update(support_switches=["s12", "s10"], initial_action=3)
        data["controllers"][2].update(support_switches=[], initial_action=0)
        network = replace(network, data=data)
        encoder = ObservationEncoder(network.controller_ids, 1, 1)
        policy = MagicMock()
        policy.decide.return_value = [2, 1, 0]

        def trial(environment, decide, log, **kwargs):
            for _ in range(kwargs["steps"]):
                kwargs["on_transition"]({"action": decide({})})

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch("EAGLE.src.evaluation.switches.EVAL", Path(temporary)),
                patch("EAGLE.src.evaluation.switches.load_network", return_value=network),
                patch(
                    "EAGLE.src.evaluation.switches.load_dataset",
                    return_value=Dataset(Path("dc_t0.json"), 100, [], "data"),
                ),
                patch(
                    "EAGLE.src.evaluation.policy_comparison.load_inference_model",
                    return_value=(policy, encoder),
                ),
                patch(
                    "EAGLE.src.evaluation.switches.checkpoint_digest",
                    return_value="digest",
                ),
                patch("EAGLE.src.evaluation.switches.make_environment"),
                patch("EAGLE.src.evaluation.switches.RunLog"),
                patch("EAGLE.src.evaluation.switches.run_trial", side_effect=trial),
            ):
                report = evaluate_switches(self.arguments())
        self.assertEqual(report["enabled_switches"]["baseline"], [3, 3, 3])
        self.assertEqual(report["enabled_switches"]["first"], [2, 2, 2])
        self.assertEqual(report["action_sizes"], [4, 2, 1])

    def test_trial_honors_steps_beyond_dataset_horizon(self) -> None:
        environment = MagicMock()
        environment.settings = load_settings()
        observation = {"global_obs": np.zeros(12)}
        environment.reset.return_value = observation, {}
        info = {
            "step_seconds": 5,
            "raw_embedding": [0.0] * 12,
            "reward_components": [{"power_watts": 1, "total_delivered_bps": 0}],
        }
        environment.step.side_effect = [
            (observation, np.ones(3), False, final, info)
            for final in (False, False, True)
        ]
        dataset = Dataset(Path("dc_t0.json"), 5, [], "traffic-hash")
        with patch("EAGLE.src.experiments.trials.collect_flow_records", return_value=[]):
            result = run_trial(
                environment,
                lambda _: [1, 0, 1],
                MagicMock(),
                episode=1,
                dataset=dataset,
                steps=3,
                measure_flows=False,
            )
        self.assertEqual(environment.step.call_count, 3)
        self.assertEqual(result["cumulative_reward"], 3)
        environment.reset.assert_called_once_with(dataset, measure_flows=False, steps=3)
        environment.commander.finish_traffic.assert_called_once_with(5)

    def test_reset_uses_explicit_horizon_and_preserves_dataset_default(self) -> None:
        network = self.network()
        encoder = ObservationEncoder(network.controller_ids, 1, 1)
        environment = JointEnvironment(
            network, load_settings(), MagicMock(), MagicMock(), encoder
        )
        environment.commander.start_traffic.return_value = {"started_at": 0}
        dataset = Dataset(Path("dc_t0.json"), 5, [], "traffic-hash")
        with (
            patch.object(environment, "_apply", return_value={"c0": {"accepted_at": 0}}),
            patch.object(environment, "_wait_for_samples", return_value=[]),
            patch.object(environment, "_observe", return_value=({}, [])),
            patch("EAGLE.src.experiments.environment.time.sleep"),
        ):
            environment.reset(dataset, steps=3, measure_flows=False)
            self.assertEqual(environment.episode_steps, 3)
            self.assertNotIn("episode_steps", environment.settings)
            environment.reset(dataset, measure_flows=False)
            self.assertEqual(environment.episode_steps, 1)

    def test_cli_requires_network_and_steps_and_selects_default_models(self) -> None:
        with (
            patch("sys.argv", ["eval_switches", "--network", "v0.json", "--steps", "20"]),
            patch("EAGLE.src.evaluation.switches.evaluate_switches") as evaluate,
        ):
            main()
        args = evaluate.call_args.args[0]
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.network, "v0.json")
        self.assertEqual(args.dataset, "dc_t0")
        self.assertEqual(args.models, ["dc_t0", "dc_pcaps", "dc_full"])


if __name__ == "__main__":
    unittest.main()
