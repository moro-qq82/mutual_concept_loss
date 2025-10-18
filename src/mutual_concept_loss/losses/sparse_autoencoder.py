"""Loss components for the sparse autoencoder head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SparseAutoencoderLossConfig:
    """Configuration controlling the sparse autoencoder loss."""

    recon_weight: float = 1.0
    l1_weight: float = 1.0
    target_sparsity: float = 0.05


class SparseAutoencoderLoss(nn.Module):
    """Combines reconstruction MSE with an L1 sparsity penalty."""

    def __init__(self, config: SparseAutoencoderLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or SparseAutoencoderLossConfig()
        if self.config.recon_weight < 0 or self.config.l1_weight < 0:
            raise ValueError("weights must be non-negative")
        if not 0 <= self.config.target_sparsity <= 1:
            raise ValueError("target_sparsity must be within [0, 1]")

    def forward(
        self, reconstruction: torch.Tensor, target: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return weighted loss and monitoring metrics."""

        if reconstruction.shape != target.shape:
            raise ValueError("reconstruction and target must share the same shape")
        if latent.dim() != 2:
            raise ValueError("latent must be shaped as (B, latent_dim)")

        mse = F.mse_loss(reconstruction, target)
        mean_activation = latent.abs().mean()
        sparsity_penalty = (mean_activation - self.config.target_sparsity).abs()
        loss = self.config.recon_weight * mse + self.config.l1_weight * sparsity_penalty
        metrics = {
            "sparse_recon_loss": mse.detach(),
            "sparse_activation": mean_activation.detach(),
            "sparse_penalty": sparsity_penalty.detach(),
        }
        return loss, metrics
