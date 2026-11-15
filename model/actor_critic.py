"""EAGLE's per-agent actor, critic, and reward estimator networks."""

from __future__ import annotations

import torch
from numpy.typing import NDArray
from torch import nn


def _network(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_size),
    )


def make_actor(input_size: int) -> nn.Sequential:
    actor = _network(input_size, 2)
    for layer in actor:
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.bias)
            nn.init.uniform_(layer.weight, -0.04, 0.04)
    return actor


def make_value_network(input_size: int) -> nn.Sequential:
    network = _network(input_size, 1)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    return network


class ActorCritic(nn.Module):
    """A checkpoint-compatible container for the slot-ordered inference models."""

    def __init__(self, observation_size: int, agent_count: int) -> None:
        super().__init__()
        self.actor = nn.ModuleList(
            make_actor(observation_size) for _ in range(agent_count)
        )
        self.critic = nn.ModuleList(
            make_value_network(observation_size) for _ in range(agent_count)
        )

    @torch.inference_mode()
    def decide(self, observation: dict[str, NDArray]) -> NDArray:
        """Return one greedy binary action per slot from global_obs [features]."""
        data = torch.as_tensor(observation["global_obs"], dtype=torch.float32)
        return torch.stack([actor(data).argmax(dim=-1) for actor in self.actor]).numpy()
