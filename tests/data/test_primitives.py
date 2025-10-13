"""Unit tests for primitive grid operations."""

from __future__ import annotations

import torch

from mutual_concept_loss.data.primitives import (
    ColorSwapParams,
    FillComponentParams,
    RotationParams,
    ShiftParams,
    _apply_color_swap,
    _apply_fill_component,
    _apply_rotation,
    _apply_shift,
)


def _simple_grid() -> torch.Tensor:
    grid = torch.zeros((4, 4, 3), dtype=torch.float32)
    grid[..., 0] = 1.0
    grid[1:3, 1:3, :] = 0
    grid[1:3, 1:3, 1] = 1.0
    return grid


def test_rotation_preserves_one_hot() -> None:
    grid = _simple_grid()
    rotated = _apply_rotation(grid, RotationParams(k=1))
    assert rotated.shape == grid.shape
    assert torch.allclose(rotated.sum(dim=-1), torch.ones_like(rotated[..., 0]))


def test_color_swap_changes_channels() -> None:
    grid = _simple_grid()
    swapped = _apply_color_swap(grid, ColorSwapParams(source=1, target=2))
    assert torch.all(swapped[1:3, 1:3, 2] == 1)
    assert torch.all(swapped[1:3, 1:3, 1] == 0)


def test_fill_component_replaces_largest_region() -> None:
    grid = _simple_grid()
    params = FillComponentParams(source_color=1, fill_color=2)
    painted = _apply_fill_component(grid, params)
    assert torch.all(painted[1:3, 1:3, 2] == 1)
    assert torch.all(painted[1:3, 1:3, 1] == 0)


def test_shift_preserves_shape() -> None:
    grid = _simple_grid()
    shifted = _apply_shift(grid, ShiftParams(dx=1, dy=0))
    assert shifted.shape == grid.shape

