"""Read native EAGLE checkpoints or convert legacy slot-indexed model states."""

import os
from pathlib import Path
import tempfile

import torch

from .actor_critic import ActorCritic
from .observations import ObservationEncoder


def load_checkpoint(path):
    """Load tensor and primitive state on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return checkpoint


def save_checkpoint(path, data):
    """Write the complete episode-boundary state before replacing the destination."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temp = Path(stream.name)
    try:
        torch.save(data, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_inference_model(path, per_agent, scenario, encoder, *, require_native=False):
    """Load one native/legacy model or select a legacy file for each logical slot."""
    slots = scenario.controller_ids
    if path is not None:
        checkpoint = load_checkpoint(path)
        if checkpoint.get("format_version") == 1:
            saved_encoder = ObservationEncoder.from_dict(checkpoint["encoder"])
            if saved_encoder.controller_ids != slots:
                raise ValueError("Checkpoint agent order differs from the scenario")
            if saved_encoder.legacy != encoder.legacy:
                raise ValueError("Use the checkpoint's active/configured observation profile")
            agent_states = checkpoint["agents"]
            if agent_states.keys() != set(slots):
                raise ValueError("Checkpoint is missing a logical agent")
            model = ActorCritic(saved_encoder.size, len(slots))
            for index, slot in enumerate(slots):
                model.actor[index].load_state_dict(agent_states[slot]["actor"])
                model.critic[index].load_state_dict(agent_states[slot]["critic"])
            return model.eval(), saved_encoder
        if require_native:
            raise ValueError("A legacy checkpoint needs explicit source normalization bounds")
        sources = {slot: checkpoint for slot in slots}
        _check_legacy_keys(_legacy_model_state(checkpoint), slots)
    else:
        if per_agent.keys() != set(slots):
            raise ValueError("Provide a legacy checkpoint path for each logical controller")
        loaded = {file: load_checkpoint(file) for file in set(per_agent.values())}
        sources = {slot: loaded[file] for slot, file in per_agent.items()}

    model = ActorCritic(encoder.size, len(slots))
    for index, slot in enumerate(slots):
        state = _legacy_model_state(sources[slot])
        for section in ("actor", "critic"):
            layers = getattr(model, section)[index]
            prefix = "{}.{}.".format(section, index)
            picked = {key[len(prefix):]: tensor for key, tensor in state.items() if key.startswith(prefix)}
            layers.load_state_dict(picked, strict=True)
    return model.eval(), encoder


def _legacy_model_state(checkpoint):
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or not all(isinstance(name, str) for name in state):
        raise ValueError("Legacy checkpoint must contain a model state dictionary")
    return state


def _check_legacy_keys(state, slots):
    indices = {str(index) for index in range(len(slots))}
    for key in state:
        parts = key.split(".")
        if len(parts) != 4 or parts[0] not in ("actor", "critic") or parts[1] not in indices:
            raise ValueError("Unexpected legacy model key: " + key)
