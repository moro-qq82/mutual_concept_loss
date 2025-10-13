"""Helper functions for manipulating 8x8 color grids."""

from __future__ import annotations

from typing import Tuple

import torch

GridTensor = torch.Tensor


def indices_to_one_hot(indices: torch.Tensor, num_classes: int) -> GridTensor:
    """Convert an index grid to one-hot encoding along the last dimension."""

    if indices.ndim != 2:
        raise ValueError("indices must be a 2D tensor")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    h, w = indices.shape
    one_hot = torch.zeros((h, w, num_classes), dtype=torch.float32, device=indices.device)
    one_hot.scatter_(-1, indices.unsqueeze(-1).long(), 1.0)
    return one_hot


def one_hot_to_indices(one_hot: GridTensor) -> torch.Tensor:
    """Convert a one-hot encoded grid back to integer indices."""

    if one_hot.ndim != 3:
        raise ValueError("one_hot must be a 3D tensor")
    if not torch.all((one_hot == 0) | (one_hot == 1)):
        raise ValueError("one_hot grid must contain binary values")
    return one_hot.argmax(dim=-1)


def ensure_one_hot(grid: GridTensor) -> GridTensor:
    """Ensure the grid is one-hot encoded by normalizing along the color axis."""

    if grid.ndim != 3:
        raise ValueError("grid must be a 3D tensor")
    normalized = torch.zeros_like(grid)
    indices = grid.argmax(dim=-1)
    normalized.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return normalized


def clamp_coordinates(x: int, y: int, grid_size: int) -> Tuple[int, int]:
    """Clamp the coordinates to lie inside the grid."""

    x = max(0, min(grid_size - 1, x))
    y = max(0, min(grid_size - 1, y))
    return x, y

