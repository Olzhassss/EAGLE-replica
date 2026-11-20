"""Exercise manual-traffic inference without Mininet or Ryu processes."""

import argparse
import copy
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from model.controller_client import ControllerUnavailableError, JointActionError
from model.interactive import InteractiveSimulation, main, run_interactive
from model.observations import ObservationEncoder
from model.scenario import ROOT, load_scenario
from model.settings import DEFAULT_SETTINGS


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.network = load_scenario(ROOT / "network_preparation/source_v0.json")
        self.encoder = ObservationEncoder(self.network.controller_ids, 1, 1)
        self.policy = MagicMock()
        self.policy.decide.return_value = np.array([0, 1, 0])
        self.client = MagicMock()
        self.simulation = InteractiveSimulation(
            self.network,
            self.policy,
            self.encoder,
            self.client,
            5,
            15,
        )

    def samples(self, actions=(1, 1, 1), sampled_at=None):
        sampled_at = time.time() if sampled_at is None else sampled_at
        resources = {
            slot: {"action": action, "revision": [1, slot]}
            for slot, action in zip(self.network.controller_ids, actions, strict=True)
        }
        return {
            row["id"]: {
                "sampled_at": sampled_at,
                "window_start": sampled_at - 1,
                "resources": copy.deepcopy(resources),
                "topology": {
                    "nodes": [{"id": row["support_switch"]}]
                    if resources[row["id"]]["action"]
                    else [],
                    "links": [],
                },
                "usage": {"switch_egress_bps": {row["support_switch"]: 2_000_000}},
            }
            for row in self.network.data["controllers"]
        }

    def arguments(self, **updates):
        return argparse.Namespace(
            **dict(
                {
                    "network": "v0.json",
                    "model": "dc_full",
                    "device": "cpu",
                    "settings": DEFAULT_SETTINGS,
                    "mininet_socket": None,
                    "decision_seconds": None,
                },
                **updates,
            )
        )

    def test_reads_slot_ordered_observations_and_only_sends_policy_actions(self):
        samples = self.samples()
        self.client.fetch_stats.side_effect = samples.__getitem__
        acknowledgments = {
            slot: {"action": action, "revision": [2, slot], "accepted_at": time.time()}
            for slot, action in zip(self.network.controller_ids, [0, 1, 0], strict=True)
        }

        def apply(actions, deadline):
            self.simulation.stop.set()
            return acknowledgments

        self.client.apply_actions.side_effect = apply
        self.simulation.run()
        self.assertIsNone(self.simulation.error)
        self.assertEqual(self.simulation.decisions, 1)
        self.assertEqual(
            self.client.apply_actions.call_args.args[0], {"c0": 0, "c1": 1, "c2": 0}
        )
        raw = np.concatenate(
            [
                self.encoder.embed_local(samples[slot]["topology"])
                for slot in self.network.controller_ids
            ]
        )
        np.testing.assert_array_equal(
            self.policy.decide.call_args.args[0]["global_obs"],
            self.encoder.normalize(raw),
        )

    def test_rejects_stale_inconsistent_and_pre_action_windows(self):
        rows = list(self.samples().values())
        self.assertTrue(self.simulation._ready(rows, {}, 0))
        self.assertFalse(self.simulation._ready(rows[:-1], {}, 0))
        self.assertFalse(
            self.simulation._ready(list(self.samples(sampled_at=0).values()), {}, 0)
        )
        self.assertFalse(self.simulation._ready(rows, {}, time.time()))
        ack = {"c0": {"action": 0, "revision": [2, "c0"]}}
        self.assertFalse(self.simulation._ready(rows, ack, 0))
        rows[-1]["resources"]["c0"]["revision"] = [2, "c0"]
        self.assertFalse(self.simulation._ready(rows, {}, 0))

    def test_status_names_observed_nodes_and_marks_missing_or_old_data(self):
        self.assertIn("s12 (c0)=WAITING", self.simulation.status())
        self.simulation.samples = self.samples((0, 1, 0))
        self.simulation.unavailable.clear()
        status = self.simulation.status()
        for value in ("s12 (c0)=OFF", "s11 (c1)=ON", "s13 (c2)=OFF", "2.0 Mb/s"):
            self.assertIn(value, status)
        self.simulation.samples["c1"]["topology"]["nodes"] = []
        self.assertIn("s11 (c1)=UNAVAILABLE", self.simulation.status())
        self.simulation.unavailable.add("c0")
        self.simulation.samples["c2"]["sampled_at"] = 0
        status = self.simulation.status()
        self.assertIn("s12 (c0)=WAITING (last OFF)", status)
        self.assertIn("s13 (c2)=STALE (last OFF)", status)

    def test_display_remains_available_while_action_request_blocks(self):
        samples = self.samples()
        self.client.fetch_stats.side_effect = samples.__getitem__
        applying, release = threading.Event(), threading.Event()

        def apply(actions, deadline):
            applying.set()
            release.wait(2)
            raise JointActionError({}, {"c0": "fixture failure"})

        self.client.apply_actions.side_effect = apply
        worker = threading.Thread(target=self.simulation.run)
        worker.start()
        try:
            self.assertTrue(applying.wait(2))
            self.assertIn("applying policy actions", self.simulation.status())
            self.assertFalse(self.simulation.done.is_set())
        finally:
            release.set()
            worker.join(2)
        self.assertTrue(self.simulation.done.is_set())
        self.assertIsInstance(self.simulation.error, JointActionError)

    def test_unavailable_telemetry_times_out_without_applying_actions(self):
        self.client.fetch_stats.side_effect = ControllerUnavailableError(
            "fixture offline"
        )
        self.simulation.sample_timeout = 0
        self.simulation.run()
        self.client.apply_actions.assert_not_called()
        self.assertIsInstance(self.simulation.error, ControllerUnavailableError)

    def test_decision_cadence_and_post_action_wait_are_independent_of_polling(self):
        clock = [100.0]
        requests = []

        def fetch(slot):
            # The first interval following an action still belongs to the old window.
            sample = self.samples(sampled_at=clock[0])[slot]
            sample["window_start"] = clock[0] - 2
            return sample

        def apply(actions, deadline):
            requests.append(clock[0])
            return {
                slot: {"action": action, "revision": [1, slot], "accepted_at": clock[0]}
                for slot, action in actions.items()
            }

        def wait(seconds):
            clock[0] += seconds
            if clock[0] > 111:
                self.simulation.stop.set()

        self.policy.decide.return_value = np.array([1, 1, 1])
        self.client.fetch_stats.side_effect = fetch
        self.client.apply_actions.side_effect = apply
        with (
            patch("model.interactive.time.monotonic", side_effect=lambda: clock[0]),
            patch("model.interactive.time.time", side_effect=lambda: clock[0]),
            patch.object(self.simulation.stop, "wait", side_effect=wait),
        ):
            self.simulation.run()
        self.assertIsNone(self.simulation.error)
        self.assertEqual(requests, [100, 105, 110])
        self.assertEqual(self.client.fetch_stats.call_count, 12 * 3)

        clock[0] = 100
        requests.clear()
        self.simulation.stop.clear()
        self.simulation.decision_seconds = 1
        with (
            patch("model.interactive.time.monotonic", side_effect=lambda: clock[0]),
            patch("model.interactive.time.time", side_effect=lambda: clock[0]),
            patch.object(self.simulation.stop, "wait", side_effect=wait),
        ):
            self.simulation.run()
        self.assertEqual(requests, [100, 102, 104, 106, 108, 110])

    def test_startup_checks_model_and_mininet_without_resetting_traffic(self):
        with (
            patch("model.interactive.load_network", return_value=self.network),
            patch(
                "model.interactive.load_inference_model",
                return_value=(self.policy, self.encoder),
            ),
            patch("model.interactive.Emulation") as emulation,
            patch(
                "model.interactive.unix_command", return_value={"scenario": "wrong"}
            ) as command,
            patch("model.interactive.ControllerClient") as client,
        ):
            with self.assertRaisesRegex(ValueError, "Mininet network differs"):
                run_interactive(self.arguments())
        command.assert_called_once_with(emulation.return_value.mininet_socket, "status")
        emulation.return_value.reset.assert_not_called()
        emulation.return_value.start_traffic.assert_not_called()
        emulation.return_value.finish_traffic.assert_not_called()
        client.assert_not_called()

    def test_invalid_cadence_fails_before_runtime_access(self):
        with patch("model.interactive.Emulation") as emulation:
            for value in (0, -1, float("nan")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_interactive(self.arguments(decision_seconds=value))
        emulation.assert_not_called()

    def test_ctrl_c_stops_worker_without_resetting_or_finishing_manual_traffic(self):
        simulation = MagicMock()
        simulation.error = None
        simulation.done.is_set.return_value = False
        simulation.done.wait.side_effect = KeyboardInterrupt
        with (
            patch("model.interactive.load_network", return_value=self.network),
            patch(
                "model.interactive.load_inference_model",
                return_value=(self.policy, self.encoder),
            ),
            patch("model.interactive.Emulation") as emulation,
            patch(
                "model.interactive.unix_command",
                return_value={"scenario": self.network.fingerprint},
            ) as command,
            patch("model.interactive.ControllerClient"),
            patch("model.interactive.InteractiveSimulation", return_value=simulation),
            patch("model.interactive.threading.Thread") as thread,
            patch("builtins.print"),
        ):
            run_interactive(self.arguments())
        command.assert_called_once_with(emulation.return_value.mininet_socket, "status")
        emulation.return_value.reset.assert_not_called()
        emulation.return_value.start_traffic.assert_not_called()
        emulation.return_value.finish_traffic.assert_not_called()
        simulation.status.assert_called_once()
        simulation.stop.set.assert_called_once()
        thread.return_value.join.assert_called_once()

    def test_cli_selects_one_model_and_defaults_to_v0(self):
        with (
            patch("sys.argv", ["interactive", "--model", "dc_full"]),
            patch("model.interactive.run_interactive") as run,
        ):
            main()
        self.assertEqual(run.call_args.args[0].model, "dc_full")
        self.assertEqual(run.call_args.args[0].network, "v0.json")
        self.assertIsNone(run.call_args.args[0].decision_seconds)


if __name__ == "__main__":
    unittest.main()
