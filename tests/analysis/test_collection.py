from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from mutual_concept_loss.analysis import (
    RepresentationCollection,
    collect_representations,
    group_by_label,
)
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


def test_collect_representations_limit_batches_and_no_flatten() -> None:
    class _DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "representation": inputs + 1.0,
                "sparse_code": inputs.mean(dim=-1, keepdim=True),
            }

    model = _DummyModel().train()
    dataset = SyntheticTaskDataset(num_samples=6)
    dataloader = DataLoader(dataset, batch_size=2)
    collection = collect_representations(
        model,
        dataloader,
        device="cpu",
        keys=("representation", "sparse_code"),
        flatten=False,
        limit_batches=1,
    )
    assert collection.features["representation"].shape == (2, 8, 8, 4)
    assert collection.features["sparse_code"].shape == (2, 8, 8, 1)
    assert collection.primitives.shape[0] == 2
    assert collection.sequence_length.shape[0] == 2
    assert collection.primitive_indices is not None
    assert collection.primitive_indices.shape[0] == 2
    assert model.training, "collect_representations should restore training mode"


def test_collect_representations_raises_when_key_missing() -> None:
    class _BadModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"representation": inputs}

    model = _BadModel()
    dataset = SyntheticTaskDataset(num_samples=2)
    dataloader = DataLoader(dataset, batch_size=1)
    with pytest.raises(KeyError):
        collect_representations(
            model,
            dataloader,
            device="cpu",
            keys=("representation", "sparse_code"),
        )


def test_representation_collection_to_creates_device_copy() -> None:
    collection = RepresentationCollection(
        features={"representation": torch.randn(2, 3)},
        primitives=torch.randn(2, 4),
        sequence_length=torch.tensor([1, 2]),
        primitive_indices=torch.tensor([[0, 1], [1, -1]]),
    )
    moved = collection.to("cpu")
    assert moved is not collection
    assert moved.features.keys() == collection.features.keys()
    assert all(t.device.type == "cpu" for t in moved.features.values())
    assert moved.primitives.device.type == "cpu"
    assert moved.sequence_length.device.type == "cpu"
    assert moved.primitive_indices is not None
    assert moved.primitive_indices.device.type == "cpu"


def test_group_by_label_returns_expected_groups() -> None:
    tensor = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([1, 1, 2])
    groups = group_by_label(tensor, labels)
    assert set(groups.keys()) == {"1", "2"}
    assert groups["1"].shape == (2, 2)
    assert groups["2"].shape == (1, 2)
