"""Utilities for gathering model representations from dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass
class RepresentationCollection:
    """Container holding representations and accompanying metadata."""

    features: Dict[str, torch.Tensor]
    primitives: torch.Tensor
    sequence_length: torch.Tensor
    primitive_indices: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "RepresentationCollection":
        """Return a copy of the collection placed on the specified device."""

        device = torch.device(device)
        features = {key: value.to(device) for key, value in self.features.items()}
        primitives = self.primitives.to(device)
        sequence_length = self.sequence_length.to(device)
        primitive_indices = None
        if self.primitive_indices is not None:
            primitive_indices = self.primitive_indices.to(device)
        return RepresentationCollection(features, primitives, sequence_length, primitive_indices)


def collect_representations(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: torch.device | str,
    keys: Sequence[str] = ("representation", "sparse_code"),
    flatten: bool = True,
    limit_batches: int | None = None,
) -> RepresentationCollection:
    """Run inference and gather representations for downstream analysis."""

    torch_device = torch.device(device)
    model_was_training = model.training
    model.eval()
    features: Dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    primitive_labels: list[torch.Tensor] = []
    sequence_lengths: list[torch.Tensor] = []
    primitive_indices: list[torch.Tensor] = []
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if limit_batches is not None and processed >= limit_batches:
                break
            inputs = batch["input"].to(torch_device)
            outputs = model(inputs)
            batch_size = inputs.shape[0]
            for key in keys:
                tensor = outputs.get(key)
                if tensor is None:
                    raise KeyError(f"Model output did not contain key '{key}'")
                if flatten:
                    tensor = tensor.reshape(batch_size, -1)
                features[key].append(tensor.cpu())
            primitive_labels.append(batch["primitives"].to(torch.float32).cpu())
            sequence_lengths.append(batch.get("sequence_length", torch.zeros(batch_size, dtype=torch.int64)).cpu())
            indices = batch.get("primitive_indices")
            if indices is not None:
                primitive_indices.append(indices.cpu())
            processed += 1
    if model_was_training:
        model.train()
    stacked_features = {key: torch.cat(value, dim=0) if value else torch.empty(0) for key, value in features.items()}
    primitives = torch.cat(primitive_labels, dim=0) if primitive_labels else torch.empty((0, 0))
    sequence = torch.cat(sequence_lengths, dim=0) if sequence_lengths else torch.empty((0,), dtype=torch.int64)
    primitive_idx: torch.Tensor | None = None
    if primitive_indices:
        primitive_idx = pad_sequence(primitive_indices, batch_first=True, padding_value=-1)
    return RepresentationCollection(stacked_features, primitives, sequence, primitive_idx)


def group_by_label(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_names: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Group rows of ``tensor`` based on label identifiers."""

    if tensor.ndim != 2:
        raise ValueError("tensor must be 2D")
    if tensor.shape[0] != labels.shape[0]:
        raise ValueError("tensor and labels must share the first dimension")
    unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    groups: Dict[str, torch.Tensor] = {}
    for idx, label in enumerate(unique_labels):
        mask = inverse == idx
        group_tensor = tensor[mask]
        if label_names is not None and 0 <= int(label) < len(label_names):
            key = label_names[int(label)]
        else:
            key = str(int(label))
        groups[key] = group_tensor
    return groups
