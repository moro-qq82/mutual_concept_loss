"""Tests for the grouped batch sampler."""

from __future__ import annotations

from typing import Dict, List

import torch

from mutual_concept_loss.data.sampler import GroupedBatchSampler


class _DummyDataset:
    def __init__(self, labels: List[List[float]]) -> None:
        self._samples = [torch.tensor(label, dtype=torch.float32) for label in labels]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"primitives": self._samples[index]}


def test_grouped_batch_sampler_groups_identical_labels() -> None:
    dataset = _DummyDataset([[1, 0], [1, 0], [0, 1], [0, 1]])
    sampler = GroupedBatchSampler(dataset, batch_size=2)
    batches = list(iter(sampler))
    assert len(batches) == 2
    for batch in batches:
        labels = [dataset[idx]["primitives"] for idx in batch]
        assert all(torch.equal(labels[0], lbl) for lbl in labels)


def test_grouped_batch_sampler_len_matches_batches() -> None:
    dataset = _DummyDataset([[1, 0], [1, 0], [0, 1]])
    sampler = GroupedBatchSampler(dataset, batch_size=2, drop_last=False)
    assert len(sampler) == 2

