"""Adapter modules that can be inserted around linear layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AdapterConfig:
    """Configuration for linear adapters."""

    input_dim: int = 128
    bottleneck_dim: int = 32
    activation: str = "relu"


class LinearAdapter(nn.Module):
    """A simple adapter block with residual connection."""

    def __init__(self, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdapterConfig()
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.config.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.down = nn.Linear(self.config.input_dim, self.config.bottleneck_dim, bias=False)
        self.up = nn.Linear(self.config.bottleneck_dim, self.config.input_dim, bias=False)
        self.activation = nn.ReLU() if self.config.activation == "relu" else nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return inputs plus low-rank adaptation."""

        if inputs.dim() != 2 or inputs.size(-1) != self.config.input_dim:
            raise ValueError("inputs must be shaped as (B, input_dim)")
        update = self.up(self.activation(self.down(inputs)))
        return inputs + update
