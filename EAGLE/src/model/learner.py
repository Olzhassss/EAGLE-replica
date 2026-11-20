"""Local average-reward actor-critic updates and per-agent checkpoint state."""

from typing import Any

import torch
from network_schema import resolve_support_counts
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.nn import functional as F

from EAGLE.src.model.actor_critic import make_actor, make_value_network
from EAGLE.src.model.consensus import mix_contributions, pack_contribution
from EAGLE.src.model.observations import ObservationEncoder


def _cpu_state(value: Any) -> Any:
    """Copy nested checkpoint tensors to CPU while preserving container structure."""
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


class AgentLearner:
    """A single slot actor, critic, reward estimator, optimizer state, and action RNG."""

    def __init__(
        self,
        slot: str,
        encoder: ObservationEncoder,
        settings: dict[str, Any],
        seed: int,
        *,
        support_counts: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize per-slot networks, optimizers, and a reproducible CPU action RNG."""
        self.encoder = encoder
        self.index = encoder.controller_ids.index(slot)
        self.support_counts = resolve_support_counts(
            support_counts, len(encoder.controller_ids)
        )
        self.settings = settings
        self.device = torch.device(device)
        torch.manual_seed(seed + self.index)
        # Preserve the action RNG format across CPU and CUDA resumes.
        self.generator = torch.Generator().manual_seed(seed + 1_000 + self.index)
        self.actor = make_actor(encoder.size, 1 << self.support_counts[self.index]).to(
            self.device
        )
        self.critic = make_value_network(encoder.size).to(self.device)
        self.estimator = make_value_network(
            encoder.size + sum(self.support_counts),
        ).to(self.device)
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=settings["actor_lr"]),
            "critic": torch.optim.Adam(
                self.critic.parameters(),
                lr=settings["critic_lr"],
            ),
            "estimator": torch.optim.Adam(
                self.estimator.parameters(),
                lr=settings["estimator_lr"],
            ),
        }
        self.mu = 0.0

    def embed(self, topology: dict[str, Any]) -> NDArray:
        return self.encoder.embed_local(topology)

    @torch.inference_mode()
    def decide(self, observation: list[float] | NDArray) -> int:
        """Sample one action for this slot from its categorical actor."""
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        probabilities = self.actor(state).softmax(dim=-1).cpu()
        return int(torch.multinomial(probabilities, 1, generator=self.generator).item())

    def learn(self, mode: str, batch: dict[str, list[Any]]) -> dict[str, float]:
        """Update on a batch of (state, joint action, rewards, next state, terminated)."""
        states = torch.as_tensor(
            batch["state"],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            batch["next_state"],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(batch["action"], device=self.device)
        limits = torch.tensor(
            [1 << count for count in self.support_counts], device=self.device
        )
        if (
            actions.is_floating_point()
            or actions.is_complex()
            or actions.dtype == torch.bool
            or actions.ndim != 2
            or actions.shape != (len(states), len(self.support_counts))
            or (actions < 0).any()
            or (actions >= limits).any()
        ):
            raise ValueError(
                "Training actions must be integer support-switch masks in slot order"
            )
        actions = actions.long()
        rewards = torch.as_tensor(
            batch["reward"],
            dtype=torch.float32,
            device=self.device,
        )[:, self.index]
        bootstrap = 1 - torch.as_tensor(
            batch["terminated"],
            dtype=torch.float32,
            device=self.device,
        )
        bits = torch.cat(
            [
                (actions[:, index : index + 1] >> torch.arange(count, device=self.device))
                & 1
                for index, count in enumerate(self.support_counts)
            ],
            dim=-1,
        )
        estimator_input = torch.cat((states, bits.float()), dim=-1)
        metrics = {}
        if mode == "pretrain":
            self.estimator.train()
            estimate = self.estimator(estimator_input).squeeze(-1)
            metrics["estimator_loss"] = self._optimize(
                "estimator",
                F.mse_loss(estimate, rewards),
            )
            return metrics
        self.estimator.eval()
        with torch.no_grad():
            predicted_reward = self.estimator(estimator_input).squeeze(-1)
        value = self.critic(states).squeeze(-1)
        with torch.no_grad():
            future = bootstrap * self.critic(next_states).squeeze(-1)
            target = rewards - self.mu + future
            advantage = predicted_reward - self.mu + future - value
        metrics["critic_loss"] = self._optimize("critic", F.mse_loss(value, target))
        policy = Categorical(logits=self.actor(states))
        metrics["actor_loss"] = self._optimize(
            "actor",
            -(policy.log_prob(actions[:, self.index]) * advantage).mean(),
        )
        beta = self.settings["critic_lr"]
        self.mu = (1 - beta) * self.mu + beta * rewards.mean().item()
        metrics["mu"] = self.mu
        return metrics

    def _optimize(self, component: str, loss: torch.Tensor) -> float:
        """Apply one gradient update to the selected network and return its loss."""
        optimizer = self.optimizers[component]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _shared_components(self, mode: str) -> dict[str, torch.nn.Module]:
        if mode == "pretrain":
            return {"estimator": self.estimator}
        return {"critic": self.critic}

    def contribution(self, mode: str, round_number: int, fraction: float) -> bytes:
        """Serialize the estimator or critic state shared in a consensus round."""
        return pack_contribution(
            round_number,
            self._shared_components(mode),
            self.mu if mode != "pretrain" else None,
            fraction,
        )

    def mix(
        self,
        mode: str,
        round_number: int,
        local_weight: float,
        peers: list[tuple[float, bytes]],
    ) -> None:
        """Mix shared network parameters and the average-reward estimate with peers."""
        updated = mix_contributions(
            round_number,
            self._shared_components(mode),
            self.mu if mode != "pretrain" else None,
            local_weight,
            peers,
        )
        if updated is not None:
            self.mu = updated

    def state_dict(self) -> dict[str, Any]:
        """Export network, optimizer, reward, and RNG state for episode resumption."""
        state = {
            "support_counts": self.support_counts,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "estimator": self.estimator.state_dict(),
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            "mu": self.mu,
            "action_rng": self.generator.get_state(),
        }
        return state if self.device.type == "cpu" else _cpu_state(state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the complete learner state, including stochastic action sampling."""
        if tuple(state.get("support_counts", ())) != self.support_counts:
            raise ValueError("Checkpoint support-switch counts differ from the learner")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.estimator.load_state_dict(state["estimator"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        self.mu = state["mu"]
        self.generator.set_state(state["action_rng"])
