"""Tests for adapter modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mutual_concept_loss.models.adapters import AdapterConfig, LinearAdapter


def test_linear_adapter_residual_behaviour() -> None:
    config = AdapterConfig(input_dim=6, bottleneck_dim=3, activation="gelu")
    adapter = LinearAdapter(config)
    assert isinstance(adapter.activation, nn.GELU)
    inputs = torch.randn(2, 6)
    outputs = adapter(inputs)
    assert outputs.shape == inputs.shape
    assert not torch.equal(outputs, inputs)


def test_linear_adapter_validates_input_shape() -> None:
    adapter = LinearAdapter(AdapterConfig(input_dim=4, bottleneck_dim=2))
    with pytest.raises(ValueError):
        adapter(torch.randn(4))
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 3))


def test_linear_adapter_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        LinearAdapter(AdapterConfig(input_dim=0, bottleneck_dim=2))
    with pytest.raises(ValueError):
        LinearAdapter(AdapterConfig(input_dim=4, bottleneck_dim=0))
