from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QNetwork(MLP):

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__(state_dim, n_actions, hidden_sizes)
