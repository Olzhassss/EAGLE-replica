"""EAGLE's per-agent actor, critic, and reward estimator networks."""

from collections.abc import Sequence

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


def make_actor(input_size: int, action_size: int) -> nn.Sequential:
    actor = _network(input_size, action_size)
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

    def __init__(self, observation_size: int, action_sizes: Sequence[int]) -> None:
        super().__init__()
        self.actor = nn.ModuleList(
            make_actor(observation_size, size) for size in action_sizes
        )
        self.critic = nn.ModuleList(
            make_value_network(observation_size) for _ in action_sizes
        )

    @torch.inference_mode()
    def decide(self, observation: dict[str, NDArray]) -> NDArray:
        """Return one greedy support-switch mask per slot from global_obs [features]."""
        device = next(self.parameters()).device
        data = torch.as_tensor(
            observation["global_obs"], dtype=torch.float32, device=device
        )
        actions = torch.stack([actor(data).argmax(dim=-1) for actor in self.actor])
        return actions.cpu().numpy()
