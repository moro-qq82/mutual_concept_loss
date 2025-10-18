"""Shared subspace regularization losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SharedSubspaceConfig:
    """Configuration for the shared subspace regularizer."""

    n_components: int = 4
    detach_projection: bool = True
    eps: float = 1e-6


class SharedSubspaceLoss(nn.Module):
    """Encourages bottleneck features to share local subspaces."""

    def __init__(self, config: SharedSubspaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedSubspaceConfig()
        if self.config.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.config.eps <= 0:
            raise ValueError("eps must be positive")

    def forward(
        self, representations: torch.Tensor, group_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute projection distance between group subspaces."""

        if representations.dim() != 2:
            raise ValueError("representations must be shaped as (B, D)")
        if group_ids.dim() != 1 or group_ids.size(0) != representations.size(0):
            raise ValueError("group_ids must align with representations")

        projections = self._compute_projection_matrices(representations, group_ids)
        if not projections:
            zero = representations.new_tensor(0.0)
            return zero, {"shared_alignment": zero.detach()}
        stacked = torch.stack(projections)
        mean_projection = stacked.mean(dim=0)
        distances = [(proj - mean_projection).pow(2).sum() for proj in projections]
        loss = torch.stack(distances).mean()
        return loss, {"shared_alignment": loss.detach()}

    def _compute_projection_matrices(
        self, representations: torch.Tensor, group_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return projection matrices for each group."""

        projections: list[torch.Tensor] = []
        for group in torch.unique(group_ids):
            mask = group_ids == group
            group_feats = representations[mask]
            if group_feats.size(0) < 2:
                continue
            centered = group_feats - group_feats.mean(dim=0, keepdim=True)
            cov = centered.t().matmul(centered) / (group_feats.size(0) - 1 + self.config.eps)
            _, eigvecs = torch.linalg.eigh(cov)
            comps = min(self.config.n_components, eigvecs.size(1))
            basis = eigvecs[:, -comps:]
            if self.config.detach_projection:
                basis = basis.detach()
            projection = basis @ basis.t()
            projections.append(projection)
        return projections
