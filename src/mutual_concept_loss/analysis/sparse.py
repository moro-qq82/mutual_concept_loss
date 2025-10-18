"""Sparse code analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class SparseSummary:
    """Statistics describing sparse code activations."""

    mean_activation: torch.Tensor
    active_fraction: torch.Tensor
    code_primitive_correlation: torch.Tensor

    def to_dict(self) -> Dict[str, list[float]]:
        """Convert statistics into JSON serializable lists."""

        return {
            "mean_activation": self.mean_activation.tolist(),
            "active_fraction": self.active_fraction.tolist(),
            "code_primitive_correlation": self.code_primitive_correlation.tolist(),
        }


def summarize_sparse_codes(
    codes: torch.Tensor,
    primitives: torch.Tensor,
    *,
    activation_threshold: float = 1e-3,
) -> SparseSummary:
    """Compute summary statistics for sparse code activations."""

    if codes.ndim != 2:
        raise ValueError("codes must be 2D")
    if primitives.ndim != 2:
        raise ValueError("primitives must be 2D")
    if codes.shape[0] != primitives.shape[0]:
        raise ValueError("codes and primitives must share the sample dimension")
    mean_activation = codes.abs().mean(dim=0)
    active_fraction = (codes.abs() > activation_threshold).float().mean(dim=0)
    codes_centered = codes - codes.mean(dim=0, keepdim=True)
    prim_centered = primitives - primitives.mean(dim=0, keepdim=True)
    covariance = codes_centered.T @ prim_centered / max(1, codes.shape[0] - 1)
    code_std = torch.sqrt(codes_centered.pow(2).mean(dim=0, keepdim=True))
    prim_std = torch.sqrt(prim_centered.pow(2).mean(dim=0, keepdim=True))
    denom = (code_std.T + 1e-12) * (prim_std + 1e-12)
    correlation = covariance / denom
    return SparseSummary(mean_activation, active_fraction, correlation)
