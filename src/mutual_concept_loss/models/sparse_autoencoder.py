"""Sparse autoencoder module used on top of the shared bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SparseAutoencoderConfig:
    """Configuration for the sparse autoencoder head."""

    input_dim: int = 128
    latent_dim: int = 64


class SparseAutoencoder(nn.Module):
    """Two-layer sparse autoencoder operating on bottleneck activations."""

    def __init__(self, config: SparseAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SparseAutoencoderConfig()
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.encoder = nn.Linear(self.config.input_dim, self.config.latent_dim)
        self.decoder = nn.Linear(self.config.latent_dim, self.config.input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction and latent code for the inputs."""

        if inputs.dim() != 2 or inputs.size(-1) != self.config.input_dim:
            raise ValueError("inputs must be shaped as (B, input_dim)")
        latent = torch.tanh(self.encoder(inputs))
        reconstruction = self.decoder(latent)
        return reconstruction, latent
