"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import torch

from mutual_concept_loss.data.generator import GeneratorConfig, SyntheticTaskDataset
from mutual_concept_loss.data.primitives import DEFAULT_PRIMITIVES


def test_dataset_returns_valid_sample() -> None:
    config = GeneratorConfig(
        grid_size=8,
        num_colors=4,
        min_composition_length=1,
        max_composition_length=2,
    )
    dataset = SyntheticTaskDataset(num_samples=5, config=config, seed=42, buffer_size=2)
    sample = dataset[0]
    assert sample["input"].shape == (8, 8, 4)
    assert sample["target"].shape == (8, 8, 4)
    assert sample["primitives"].shape == (len(DEFAULT_PRIMITIVES),)
    assert int(sample["sequence_length"].item()) in {1, 2}
    assert torch.allclose(sample["input"].sum(dim=-1), torch.ones(8, 8))
    assert torch.allclose(sample["target"].sum(dim=-1), torch.ones(8, 8))


def test_dataset_cache_returns_clones() -> None:
    dataset = SyntheticTaskDataset(num_samples=3, seed=123, buffer_size=1)
    first = dataset[1]
    second = dataset[1]
    assert torch.equal(first["target"], second["target"])
    assert first["target"].data_ptr() != second["target"].data_ptr()

