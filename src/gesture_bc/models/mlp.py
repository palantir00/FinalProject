"""Reactive MLP policy network: current observation -> one action."""

import torch
from torch import nn

# Network outputs: 3 numbers for the TCP movement (regression) + 1 gripper logit.
MOVE_DIM = 3
OUTPUT_DIM = MOVE_DIM + 1


class MLP(nn.Module):
    """Fully connected network with ReLU activations and dropout.

    Input:  normalized observation (obs_dim numbers).
    Output: normalized movement (3) and a gripper logit (1); logit > 0 means "close".
    The network has no memory: the same observation always gives the same action.
    """

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (256, 256, 256), dropout=0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights (reported to compare models of similar size)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
