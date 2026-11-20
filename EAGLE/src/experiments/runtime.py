"""Construct experiment dependencies and seed reproducible policy runs."""

import argparse
import random

import numpy as np
import torch

from EAGLE.src.communication.controller_client import ControllerClient
from EAGLE.src.communication.socket_commander import SocketCommander
from EAGLE.src.experiments.environment import JointEnvironment
from EAGLE.src.model.observations import ObservationEncoder
from EAGLE.src.network import Network


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before constructing a model or trial."""
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)


def make_environment(
    args: argparse.Namespace,
    network: Network,
    settings: dict,
    encoder: ObservationEncoder,
) -> JointEnvironment:
    """Connect a prepared network to the controller client and local services."""
    return JointEnvironment(
        network,
        settings,
        ControllerClient(network),
        SocketCommander(args.mininet_socket, args.ryu_socket),
        encoder,
    )
