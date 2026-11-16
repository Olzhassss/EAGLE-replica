"""Read and write native EAGLE checkpoints."""

import tempfile
from pathlib import Path
from typing import Any

import torch

from .actor_critic import ActorCritic
from .observations import ObservationEncoder
from .scenario import Scenario

def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensor and primitive state on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"encoder", "agents"}.issubset(
        checkpoint
    ):
        raise ValueError("Expected an EAGLE checkpoint with encoder and agent state")
    return checkpoint


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Write the complete episode-boundary state before replacing the destination."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix="checkpoint-",
        suffix=".pth",
        delete=False,
    ) as stream:
        temp = Path(stream.name)
    try:
        torch.save(data, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_inference_model(
    path: Path,
    scenario: Scenario,
) -> tuple[ActorCritic, ObservationEncoder]:
    """Restore slot-ordered networks and their saved observation encoder."""
    checkpoint = load_checkpoint(path)
    encoder = ObservationEncoder.from_dict(checkpoint["encoder"])
    slots = scenario.controller_ids
    if encoder.controller_ids != slots:
        raise ValueError("Checkpoint agent order differs from the scenario")
    agent_states = checkpoint["agents"]
    if agent_states.keys() != set(slots):
        raise ValueError("Checkpoint agent slots differ from the scenario")
    model = ActorCritic(encoder.size, len(slots))
    for index, slot in enumerate(slots):
        model.actor[index].load_state_dict(agent_states[slot]["actor"])
        model.critic[index].load_state_dict(agent_states[slot]["critic"])
    return model.eval(), encoder
