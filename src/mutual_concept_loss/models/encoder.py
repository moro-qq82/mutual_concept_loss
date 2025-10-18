"""Encoders that project grid inputs into latent representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncoderConfig:
    """Configuration parameters for the convolutional grid encoder."""

    grid_size: int = 8
    input_channels: int = 4
    hidden_dim: int = 128
    conv_channels: int = 64


class ConvGridEncoder(nn.Module):
    """A lightweight convolutional encoder for 8x8 one-hot grids."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.conv_channels <= 0:
            raise ValueError("conv_channels must be positive")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                self.config.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.config.conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.conv_channels,
                self.config.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.config.conv_channels),
            nn.ReLU(inplace=True),
        )
        flattened_dim = self.config.conv_channels * (self.config.grid_size**2)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grids shaped as (B, H, W, C)."""

        if inputs.dim() != 4:
            raise ValueError("inputs must be a 4D tensor")
        _, height, width, channels = inputs.shape
        if height != self.config.grid_size or width != self.config.grid_size:
            raise ValueError("input grid does not match configured grid_size")
        if channels != self.config.input_channels:
            raise ValueError("input channels do not match configuration")

        tensor = inputs.permute(0, 3, 1, 2).contiguous()
        features = self.feature_extractor(tensor)
        return self.projection(features)
