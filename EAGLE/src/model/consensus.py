"""Parameter mixing and packed L1-pruned consensus messages for trusted workers."""

import pickle
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch


def pruning_fraction(config: Mapping[str, Any], round_number: int) -> float:
    """Return the dense, static, or warmup-and-ramp sparsification fraction."""
    mode = config["mode"]
    if mode == "none":
        return 0.0
    if mode == "static":
        return 0.7
    if mode != "dynamic":
        raise ValueError("pruning.mode must be none, static, or dynamic")
    warmup = config["warmup_rounds"]
    if round_number <= warmup:
        return 0.0
    progress = min(1.0, (round_number - warmup - 1) / config["ramp_rounds"])
    return 0.3 + 0.4 * progress


def pack_contribution(
    round_number: int,
    networks: Mapping[str, torch.nn.Module],
    mu: float | None,
    fraction: float,
) -> bytes:
    """Pack copied float32 weights and masks. Keep each first layer and all biases."""
    packed = {}
    for component, model in networks.items():
        packed[component] = {}
        for name, tensor in model.state_dict().items():
            values = tensor.detach().cpu().numpy().ravel()
            eligible = name.endswith(".weight") and name != "0.weight"
            count = round(fraction * values.size) if eligible else 0
            if count:
                retained = np.ones(values.size, dtype=bool)
                retained[np.argsort(np.abs(values), kind="stable")[:count]] = False
                mask = np.packbits(retained).tobytes()
                payload = values[retained].tobytes()
            else:
                mask = None
                payload = values.tobytes()
            packed[component][name] = (tuple(tensor.shape), mask, payload)
    return pickle.dumps((round_number, packed, mu), protocol=5)


def _unpack_tensor(
    shape: tuple[int, ...],
    mask: bytes | None,
    payload: bytes,
) -> torch.Tensor:
    """Restore a dense tensor or scatter a sparse payload into its declared shape."""
    values = np.frombuffer(payload, dtype=np.float32)
    if mask is None:
        return torch.from_numpy(values.copy().reshape(shape))
    size = int(np.prod(shape))
    retained = np.unpackbits(np.frombuffer(mask, dtype=np.uint8), count=size).astype(
        bool,
    )
    dense = np.zeros(size, dtype=np.float32)
    dense[retained] = values
    return torch.from_numpy(dense.reshape(shape))


@torch.no_grad()
def mix_contributions(
    round_number: int,
    networks: Mapping[str, torch.nn.Module],
    mu: float | None,
    local_weight: float,
    peer_packets: Iterable[tuple[float, bytes]],
) -> float | None:
    """Mix unpruned local tensors with reconstructed peer contributions by name."""
    mixed = {
        component: {
            name: tensor.detach().clone().mul_(local_weight)
            for name, tensor in model.state_dict().items()
        }
        for component, model in networks.items()
    }
    new_mu = mu * local_weight if mu is not None else None
    for weight, packet in peer_packets:
        message_round, components, peer_mu = pickle.loads(packet)
        if (
            message_round != round_number
            or components.keys() != mixed.keys()
            or (peer_mu is None) != (mu is None)
        ):
            raise ValueError("Invalid consensus round or component set")
        if peer_mu is not None:
            new_mu += weight * peer_mu
        for component, tensors in components.items():
            if tensors.keys() != mixed[component].keys():
                raise ValueError("Consensus parameter keys do not match")
            for name, payload in tensors.items():
                target = mixed[component][name]
                if tuple(target.shape) != payload[0]:
                    raise ValueError("Consensus shape does not match " + name)
                target.add_(_unpack_tensor(*payload).to(target.device), alpha=weight)
    for component, model in networks.items():
        model.load_state_dict(mixed[component])
    return new_mu
