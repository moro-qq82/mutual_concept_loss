from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mutual_concept_loss.analysis import collect_representations, group_by_label
from mutual_concept_loss.data import SyntheticTaskDataset
from mutual_concept_loss.models import SharedAutoencoderModel


def test_collect_representations_basic() -> None:
    model = SharedAutoencoderModel()
    dataset = SyntheticTaskDataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=2)
    collection = collect_representations(
        model,
        dataloader,
        device="cpu",
        keys=("representation", "sparse_code"),
    )
    assert collection.features["representation"].shape[0] == 4
    assert collection.features["sparse_code"].shape[0] == 4
    assert collection.primitives.shape[0] == 4
    assert collection.sequence_length.shape[0] == 4
    assert collection.primitive_indices is not None
    assert collection.primitive_indices.shape[0] == 4


def test_group_by_label_returns_expected_groups() -> None:
    tensor = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([1, 1, 2])
    groups = group_by_label(tensor, labels)
    assert set(groups.keys()) == {"1", "2"}
    assert groups["1"].shape == (2, 2)
    assert groups["2"].shape == (1, 2)
