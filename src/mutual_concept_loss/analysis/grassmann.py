"""Grassmannian distance utilities for PCA-based subspaces."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch


def compute_principal_components(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """Return the top-k principal components of the given features.

    Uses full SVD in float64 for numerical stability so that identical
    subspaces yield zero Grassmann distance up to tight tolerances.
    """

    if tensor.ndim != 2:
        raise ValueError("tensor must be 2D")
    if k <= 0:
        raise ValueError("k must be positive")
    samples, dim = tensor.shape
    k = min(k, samples, dim)
    if k == 0:
        raise ValueError("k reduced to zero; ensure tensor has sufficient rank")

    # Center and compute SVD in float64 for better orthonormality
    centered = (tensor.to(torch.float64) - tensor.to(torch.float64).mean(dim=0, keepdim=True))
    # torch.linalg.svd returns U (n x r), S (r), Vh (r x d) with r=min(n,d)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    V = Vh.mT  # (d x r), columns are orthonormal
    # 再直交化で数値誤差を抑える
    Q, _ = torch.linalg.qr(V[:, :k], mode="reduced")
    # 安定計算にfloat64を使うが、出力は元テンソルのdtypeに戻す
    return Q.to(tensor.dtype).contiguous()


def _principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute principal angles between two orthonormal bases."""

    if basis_a.shape != basis_b.shape:
        raise ValueError("basis_a and basis_b must share the same shape")
    # まず各基底をfloat64で再直交化
    Qa = torch.linalg.qr(basis_a.to(torch.float64), mode="reduced")[0]
    Qb = torch.linalg.qr(basis_b.to(torch.float64), mode="reduced")[0]
    gram = Qa.T @ Qb
    sigma = torch.linalg.svdvals(gram)
    # 計算誤差で -1 未満に出るのを防ぐための下限のみ微小緩和。上限は 1.0 を許容する
    sigma = sigma.clamp(min=-1.0 + eps, max=1.0)
    return torch.arccos(sigma)


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
