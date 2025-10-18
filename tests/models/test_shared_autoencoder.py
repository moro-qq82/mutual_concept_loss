"""Tests for the shared autoencoder model."""

from __future__ import annotations

import torch

from mutual_concept_loss.models import (
    LinearAdapter,
    SharedAutoencoderConfig,
    SharedAutoencoderModel,
)


def test_shared_autoencoder_forward_shapes() -> None:
    model = SharedAutoencoderModel(SharedAutoencoderConfig(num_primitives=5))
    grid = torch.randn(2, 8, 8, 4)
    outputs = model(grid)
    assert outputs["task_logits"].shape == (2, 4, 8, 8)
    assert outputs["reconstruction"].shape == (2, 8, 8, 4)
    assert outputs["representation"].shape == (2, model.config.bottleneck.hidden_dim)
    assert outputs["sparse_code"].shape[1] == model.config.sparse.latent_dim
    assert outputs["primitive_logits"].shape == (2, 5)


def test_shared_autoencoder_adapter_lifecycle() -> None:
    model = SharedAutoencoderModel()
    adapter = LinearAdapter()
    model.attach_adapter(adapter)
    grid = torch.randn(1, 8, 8, 4)
    outputs = model(grid)
    assert outputs["representation"].shape[-1] == model.config.bottleneck.hidden_dim
    adapter_params = list(model.adapter_parameters())
    assert adapter_params, "adapter parameters should be exposed"
    model.detach_adapter()
    assert list(model.adapter_parameters()) == []
