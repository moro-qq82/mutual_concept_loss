"""Bottleneck modules shared across downstream heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BottleneckConfig:
    """Configuration for the shared bottleneck layer."""

    input_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.0


class SharedBottleneck(nn.Module):
    """Projects encoder features into a normalized shared space."""

    def __init__(self, config: BottleneckConfig | None = None) -> None:
        super().__init__()
        self.config = config or BottleneckConfig()
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")

        self.linear = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return bottleneck activations for downstream modules."""

        if features.dim() != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("features must be shaped as (B, input_dim)")
        projected = self.linear(features)
        normalized = self.norm(projected)
        activated = self.activation(normalized)
        return self.dropout(activated)
