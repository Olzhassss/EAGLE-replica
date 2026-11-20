"""Read and write native EAGLE checkpoints."""

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import torch

from EAGLE.src.model.actor_critic import ActorCritic
from EAGLE.src.model.observations import ObservationEncoder
from EAGLE.src.network import Network


def checkpoint_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensor and primitive state on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {
        "encoder",
        "agents",
        "support_counts",
    }.issubset(checkpoint):
        raise ValueError(
            "Expected an EAGLE checkpoint with encoder and agent state and support_counts; regenerate older checkpoints"
        )
    return checkpoint


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Write the complete episode-boundary state before replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
    network: Network,
) -> tuple[ActorCritic, ObservationEncoder]:
    """Restore slot-ordered networks and their saved observation encoder."""
    checkpoint = load_checkpoint(path)
    encoder = ObservationEncoder.from_dict(checkpoint["encoder"])
    slots = network.controller_ids
    if encoder.controller_ids != slots:
        raise ValueError("Checkpoint agent order differs from the network")
    if tuple(checkpoint["support_counts"]) != network.support_counts:
        raise ValueError("Checkpoint support-switch counts differ from the network")
    agent_states = checkpoint["agents"]
    if agent_states.keys() != set(slots):
        raise ValueError("Checkpoint agent slots differ from the network")
    model = ActorCritic(encoder.size, network.action_sizes)
    for index, slot in enumerate(slots):
        model.actor[index].load_state_dict(agent_states[slot]["actor"])
        model.critic[index].load_state_dict(agent_states[slot]["critic"])
    return model.eval(), encoder
