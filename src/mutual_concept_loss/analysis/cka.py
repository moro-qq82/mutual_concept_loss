"""Linear CKA implementation for representation comparisons."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch


def _center_features(tensor: torch.Tensor) -> torch.Tensor:
    """Return a centered version of the feature matrix."""

    if tensor.ndim != 2:
        raise ValueError("tensor must be 2D")
    return tensor - tensor.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor, *, center: bool = True, eps: float = 1e-12) -> float:
    """Compute linear CKA similarity between two feature matrices."""

    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature matrices must share the sample dimension")
    if center:
        x = _center_features(x)
        y = _center_features(y)
    cross_cov = x.T @ y
    numerator = torch.linalg.norm(cross_cov, ord="fro") ** 2
    xx = x.T @ x
    yy = y.T @ y
    denom = torch.linalg.norm(xx, ord="fro") * torch.linalg.norm(yy, ord="fro")
    return float(numerator / (denom + eps))


def pairwise_linear_cka(
    groups: Dict[str, torch.Tensor] | Sequence[torch.Tensor] | Iterable[torch.Tensor],
    *,
    labels: Sequence[str] | None = None,
    center: bool = True,
) -> Tuple[Sequence[str], torch.Tensor]:
    """Compute a pairwise CKA matrix for the provided groups."""

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
    matrix = torch.zeros((len(tensors), len(tensors)), dtype=torch.float64)
    for i, x in enumerate(tensors):
        for j in range(i, len(tensors)):
            value = linear_cka(x, tensors[j], center=center)
            matrix[i, j] = value
            matrix[j, i] = value
    return names, matrix
