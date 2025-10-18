"""High level model that combines encoder, bottleneck and sparse heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn

from .bottleneck import BottleneckConfig, SharedBottleneck
from .decoder import ConvGridDecoder, DecoderConfig
from .encoder import ConvGridEncoder, EncoderConfig
from .sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig


@dataclass
class SharedAutoencoderConfig:
    """Configuration for the high level shared autoencoder."""

    grid_size: int = 8
    num_colors: int = 4
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    sparse: SparseAutoencoderConfig = field(default_factory=SparseAutoencoderConfig)
    num_primitives: int = 8

    def __post_init__(self) -> None:
        self.encoder.grid_size = self.grid_size
        self.encoder.input_channels = self.num_colors
        self.bottleneck.input_dim = self.encoder.hidden_dim
        self.decoder.grid_size = self.grid_size
        self.decoder.output_channels = self.num_colors
        self.decoder.hidden_dim = self.bottleneck.hidden_dim
        self.sparse.input_dim = self.bottleneck.hidden_dim


class SharedAutoencoderModel(nn.Module):
    """Model that predicts task outputs and auxiliary primitive labels."""

    def __init__(self, config: SharedAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedAutoencoderConfig()
        self.encoder = ConvGridEncoder(self.config.encoder)
        self.bottleneck = SharedBottleneck(self.config.bottleneck)
        self.decoder = ConvGridDecoder(self.config.decoder)
        self.sparse = SparseAutoencoder(self.config.sparse)
        self.aux_head = nn.Linear(self.config.bottleneck.hidden_dim, self.config.num_primitives)
        self.adapter: nn.Module = nn.Identity()

    def forward(self, grid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute task logits, reconstructions, and auxiliary predictions."""

        features = self.encoder(grid)
        shared = self.bottleneck(features)
        shared = self.adapter(shared)
        grid_logits = self.decoder(shared)
        reconstruction = grid_logits.permute(0, 2, 3, 1).contiguous()
        sparse_recon, sparse_code = self.sparse(shared)
        primitive_logits = self.aux_head(shared)
        return {
            "task_logits": grid_logits,
            "reconstruction": reconstruction,
            "representation": shared,
            "sparse_reconstruction": sparse_recon,
            "sparse_code": sparse_code,
            "primitive_logits": primitive_logits,
        }

    def attach_adapter(self, adapter: nn.Module) -> None:
        """Register an adapter module applied after the bottleneck."""

        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter must be an nn.Module")
        self.adapter = adapter

    def detach_adapter(self) -> None:
        """Remove the currently registered adapter."""

        self.adapter = nn.Identity()

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        """Return parameters belonging to the active adapter."""

        return self.adapter.parameters()
