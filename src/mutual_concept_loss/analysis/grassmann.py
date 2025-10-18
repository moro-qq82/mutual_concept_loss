"""Grassmannian distance utilities for PCA-based subspaces."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch


def compute_principal_components(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """Return the top-k principal components of the given features."""

    if tensor.ndim != 2:
        raise ValueError("tensor must be 2D")
    if k <= 0:
        raise ValueError("k must be positive")
    samples, dim = tensor.shape
    k = min(k, samples, dim)
    if k == 0:
        raise ValueError("k reduced to zero; ensure tensor has sufficient rank")
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(centered, q=k)
    # torch.pca_lowrank returns right singular vectors (v) with shape (dim, k)
    return v[:, :k]


def _principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute principal angles between two orthonormal bases."""

    if basis_a.shape != basis_b.shape:
        raise ValueError("basis_a and basis_b must share the same shape")
    gram = basis_a.T @ basis_b
    sigma = torch.linalg.svdvals(gram)
    sigma = sigma.clamp(min=-1.0 + eps, max=1.0 - eps)
    return torch.arccos(sigma.clamp(-1.0, 1.0))


def grassmann_distance(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """Return the Grassmann geodesic distance between two subspaces."""

    angles = _principal_angles(basis_a, basis_b)
    return float(torch.linalg.norm(angles))


def pairwise_grassmann_distances(
    groups: Dict[str, torch.Tensor] | Sequence[torch.Tensor] | Iterable[torch.Tensor],
    *,
    labels: Sequence[str] | None = None,
    k: int,
) -> Tuple[Sequence[str], torch.Tensor]:
    """Compute pairwise Grassmann distances between PCA subspaces."""

    if isinstance(groups, dict):
        names = list(groups.keys())
        tensors = list(groups.values())
    else:
        tensors = list(groups)
        if labels is not None:
            names = list(labels)
        else:
            names = [str(i) for i in range(len(tensors))]
    if not tensors:
        return names, torch.empty((0, 0))
    bases = [compute_principal_components(tensor, k=k) for tensor in tensors]
    matrix = torch.zeros((len(bases), len(bases)), dtype=torch.float64)
    for i, basis_a in enumerate(bases):
        for j in range(i, len(bases)):
            value = grassmann_distance(basis_a, bases[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return names, matrix
