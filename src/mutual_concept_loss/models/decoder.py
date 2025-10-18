"""Decoders that reconstruct grids from latent codes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DecoderConfig:
    """Configuration for the convolutional grid decoder."""

    grid_size: int = 8
    output_channels: int = 4
    hidden_dim: int = 128
    conv_channels: int = 64


class ConvGridDecoder(nn.Module):
    """Maps bottleneck representations back to grid logits."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.conv_channels <= 0:
            raise ValueError("conv_channels must be positive")

        spatial_dim = self.config.grid_size**2
        self.linear = nn.Linear(
            self.config.hidden_dim, self.config.conv_channels * spatial_dim
        )
        self.layers = nn.Sequential(
            nn.Conv2d(
                self.config.conv_channels,
                self.config.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.config.conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.conv_channels,
                self.config.output_channels,
                kernel_size=1,
            ),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Return grid logits shaped as (B, C, H, W)."""

        if representation.dim() != 2 or representation.size(-1) != self.config.hidden_dim:
            raise ValueError("representation must be shaped as (B, hidden_dim)")
        batch = representation.size(0)
        features = self.linear(representation)
        tensor = features.view(
            batch,
            self.config.conv_channels,
            self.config.grid_size,
            self.config.grid_size,
        )
        return self.layers(tensor)
