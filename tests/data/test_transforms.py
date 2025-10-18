"""Tests for grid transform helpers."""

from __future__ import annotations

import pytest
import torch

from mutual_concept_loss.data.transforms import (
    clamp_coordinates,
    ensure_one_hot,
    indices_to_one_hot,
    one_hot_to_indices,
)


def test_indices_to_one_hot_round_trip() -> None:
    indices = torch.tensor([[0, 2], [1, 1]])
    one_hot = indices_to_one_hot(indices, num_classes=3)
    assert one_hot.shape == (2, 2, 3)
    recovered = one_hot_to_indices(one_hot)
    assert torch.equal(recovered, indices)


def test_indices_to_one_hot_validates_inputs() -> None:
    with pytest.raises(ValueError):
        indices_to_one_hot(torch.tensor([0, 1, 2]), num_classes=3)
    with pytest.raises(ValueError):
        indices_to_one_hot(torch.zeros((2, 2), dtype=torch.int64), num_classes=0)


def test_one_hot_to_indices_requires_binary_grid() -> None:
    valid = torch.zeros((2, 2, 2), dtype=torch.float32)
    valid[..., 0] = 1.0
    assert torch.equal(one_hot_to_indices(valid), torch.zeros((2, 2), dtype=torch.long))
    invalid = torch.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError):
        one_hot_to_indices(invalid)


def test_ensure_one_hot_selects_max_channel() -> None:
    grid = torch.tensor(
        [
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.9, 0.1], [0.3, 0.7]],
        ]
    )
    normalized = ensure_one_hot(grid)
    assert torch.all(normalized.sum(dim=-1) == 1)
    assert torch.equal(one_hot_to_indices(normalized), torch.tensor([[1, 0], [0, 1]]))


def test_clamp_coordinates_respects_bounds() -> None:
    assert clamp_coordinates(-1, 5, grid_size=4) == (0, 3)
    assert clamp_coordinates(10, -3, grid_size=6) == (5, 0)
