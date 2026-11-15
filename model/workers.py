"""Run learners locally or in one spawned process per logical controller."""

from contextlib import AbstractContextManager
import math
import multiprocessing
import pickle
import time
import traceback

import torch

from .consensus import pruning_fraction
from .learner import AgentLearner


def _encode(message):
    return pickle.dumps(message, protocol=5)


def _serve(connection, slot, encoder, settings, seed):
    with connection:
        torch.set_num_threads(1)
        learner = AgentLearner(slot, encoder, settings, seed)
        connection.send_bytes(b"ready")
        while True:
            try:
                operation, args = pickle.loads(connection.recv_bytes())
            except EOFError:
                return
            if operation == "stop":
                return
            try:
                reply = True, getattr(learner, operation)(*args)
            except Exception:
                reply = False, traceback.format_exc()
            connection.send_bytes(_encode(reply))


class AgentPool(AbstractContextManager):
    """Dispatch local operations and exchange contributions for a training round."""

    def __init__(self, encoder, settings, seed, *, distributed, timeout=30):
        self.slots = encoder.controller_ids
        self.encoder = encoder
        self.settings = settings
        self.timeout = timeout
        self.local = {}
        self.workers = {}
        self.bytes_transferred = 0
        context = multiprocessing.get_context("spawn")
        try:
            for slot in self.slots:
                if distributed:
                    parent, child = context.Pipe()
                    try:
                        process = context.Process(target=_serve, args=(child, slot, encoder, settings, seed), daemon=True)
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    self.workers[slot] = (parent, process)
                else:
                    self.local[slot] = AgentLearner(slot, encoder, settings, seed)
            deadline = time.monotonic() + self.timeout
            for slot, (pipe, _) in self.workers.items():
                if not pipe.poll(max(0, deadline - time.monotonic())) or pipe.recv_bytes() != b"ready":
                    raise TimeoutError("Agent {} did not initialize".format(slot))
        except BaseException:
            self.close(abort=True)
            raise

    def dispatch(self, operation, arguments):
        """Return results by slot. A worker failure aborts the round."""
        if self.local:
            return {slot: getattr(self.local[slot], operation)(*args) for slot, args in arguments.items()}
        deadline = time.monotonic() + self.timeout
        for slot, args in arguments.items():
            pipe, process = self.workers[slot]
            if process.exitcode is not None:
                raise RuntimeError("Agent {} exited during {}".format(slot, operation))
            request = _encode((operation, args))
            pipe.send_bytes(request)
            self.bytes_transferred += len(request)
        result = {}
        for slot in arguments:
            pipe, _ = self.workers[slot]
            if not pipe.poll(max(0.0, deadline - time.monotonic())):
                raise TimeoutError("Agent {} timed out during {}".format(slot, operation))
            raw = pipe.recv_bytes()
            self.bytes_transferred += len(raw)
            ok, reply = pickle.loads(raw)
            if not ok:
                raise RuntimeError("Agent {} failed during {}\n{}".format(slot, operation, reply))
            result[slot] = reply
        return result

    def embed(self, topologies):
        result = self.dispatch("embed", {slot: (graph,) for slot, graph in zip(self.slots, topologies, strict=True)})
        return [result[slot] for slot in self.slots]

    def decide(self, observation):
        row = observation["global_obs"].tolist()
        result = self.dispatch("decide", {slot: (row,) for slot in self.slots})
        return [result[slot] for slot in self.slots]

    def run_round(self, round_number, mode, batch, active_agents=None):
        if active_agents is None:
            participants = self.slots
        else:
            if not active_agents or set(active_agents) - set(self.slots) or len(set(active_agents)) != len(active_agents):
                raise ValueError("Active learning agents must be distinct scenario slots")
            participants = tuple(slot for slot in self.slots if slot in active_agents)
        rows = self.settings.get("mixing_weights", {})
        mixing = {slot: _weights_for(slot, participants, rows.get(slot)) for slot in participants}
        metrics = self.dispatch("learn", {slot: (mode, batch) for slot in participants})
        fraction = pruning_fraction(self.settings["pruning"], round_number)
        needed = {peer for slot, row in mixing.items() for peer, weight in row.items() if peer != slot and weight}
        parameters = self.dispatch("contribution", {slot: (mode, round_number, fraction)
                                                     for slot in participants if slot in needed})
        mixed_args = {}
        payload_bytes = 0
        for slot, row in mixing.items():
            peers = [(weight, parameters[peer]) for peer, weight in row.items() if peer != slot and weight]
            payload_bytes += sum(len(packet) for _, packet in peers)
            mixed_args[slot] = (mode, round_number, row.get(slot, 0.0), peers)
        self.dispatch("mix", mixed_args)
        return {"agents": metrics, "pruned_fraction": fraction, "round": round_number,
                "peer_parameter_bytes": payload_bytes}

    def state_dict(self):
        return self.dispatch("state_dict", {slot: () for slot in self.slots})

    def load_state_dict(self, state):
        if state.keys() != set(self.slots):
            raise ValueError("Checkpoint learner slots differ from the scenario")
        self.dispatch("load_state_dict", {slot: (state[slot],) for slot in self.slots})

    def close(self, *, abort=False):
        for pipe, process in self.workers.values():
            if not abort and process.is_alive():
                try:
                    pipe.send_bytes(_encode(("stop", ())))
                except OSError:
                    pass
            pipe.close()
        for _, process in self.workers.values():
            if not abort:
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

    def __exit__(self, exc_type, exc, trace):
        self.close(abort=exc_type is not None)


def _weights_for(slot, participants, configured):
    if configured is None:
        return {name: 1 / len(participants) for name in participants}
    if any(type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
           for weight in configured.values()):
        raise ValueError("Invalid consensus weights for " + slot)
    row = {name: configured.get(name, 0) for name in participants}
    total = sum(row.values())
    if not total:
        raise ValueError("No positive consensus weight among active agents for " + slot)
    return {name: weight / total for name, weight in row.items()}
