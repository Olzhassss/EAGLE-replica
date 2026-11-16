"""Local average-reward actor-critic updates and per-agent checkpoint state."""

from typing import Any

import torch
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.nn import functional as F

from .actor_critic import make_actor, make_value_network
from .consensus import mix_contributions, pack_contribution
from .observations import ObservationEncoder


def _cpu_state(value: Any) -> Any:
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
    def __init__(
        self,
        slot: str,
        encoder: ObservationEncoder,
        settings: dict[str, Any],
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.index = encoder.controller_ids.index(slot)
        self.settings = settings
        self.device = torch.device(device)
        torch.manual_seed(seed + self.index)
        # Preserve the action RNG format across CPU and CUDA resumes.
        self.generator = torch.Generator().manual_seed(seed + 1_000 + self.index)
        self.actor = make_actor(encoder.size).to(self.device)
        self.critic = make_value_network(encoder.size).to(self.device)
        self.estimator = make_value_network(
            encoder.size + len(encoder.controller_ids),
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
        actions = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
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
        estimator_input = torch.cat((states, actions.float()), dim=-1)
        self.estimator.train(mode != "frozen")
        if mode == "frozen":
            with torch.no_grad():
                estimate = self.estimator(estimator_input).squeeze(-1)
        else:
            estimate = self.estimator(estimator_input).squeeze(-1)
        predicted_reward = estimate.detach()
        metrics = {}
        if mode in ("pretrain", "online"):
            metrics["estimator_loss"] = self._optimize(
                "estimator",
                F.mse_loss(estimate, rewards),
            )
        if mode == "pretrain":
            return metrics
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
        optimizer = self.optimizers[component]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _shared_components(self, mode: str) -> dict[str, torch.nn.Module]:
        if mode == "pretrain":
            return {"estimator": self.estimator}
        if mode == "frozen":
            return {"critic": self.critic}
        return {"critic": self.critic, "estimator": self.estimator}

    def contribution(self, mode: str, round_number: int, fraction: float) -> bytes:
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
        state = {
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
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.estimator.load_state_dict(state["estimator"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        self.mu = state["mu"]
        self.generator.set_state(state["action_rng"])
