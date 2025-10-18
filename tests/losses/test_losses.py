"""Unit tests covering loss helpers."""

from __future__ import annotations

import torch

from mutual_concept_loss.losses import (
    LossManager,
    LossSchedule,
    SharedSubspaceLoss,
    SparseAutoencoderLoss,
    TaskLoss,
)


def test_task_loss_cross_entropy() -> None:
    loss_fn = TaskLoss()
    logits = torch.randn(2, 4, 8, 8)
    targets = torch.randint(0, 4, (2, 8, 8))
    loss, metrics = loss_fn(logits, targets)
    assert loss.shape == ()
    assert "task_iou" in metrics
    assert metrics["task_loss"].ndim == 0


def test_shared_subspace_loss_returns_scalar() -> None:
    reps = torch.randn(6, 16)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    loss_fn = SharedSubspaceLoss()
    loss, metrics = loss_fn(reps, groups)
    assert loss.ndim == 0
    assert metrics["shared_alignment"].ndim == 0


def test_sparse_autoencoder_loss_combines_terms() -> None:
    loss_fn = SparseAutoencoderLoss()
    reconstruction = torch.zeros(3, 10)
    target = torch.ones(3, 10)
    latent = torch.randn(3, 4)
    loss, metrics = loss_fn(reconstruction, target, latent)
    assert loss.ndim == 0
    assert metrics["sparse_recon_loss"].ndim == 0


def test_loss_manager_combines_components() -> None:
    task_loss = TaskLoss()
    shared_loss = SharedSubspaceLoss()
    sparse_loss = SparseAutoencoderLoss()
    manager = LossManager(
        task_loss,
        shared_loss,
        sparse_loss,
        alpha=LossSchedule(1.0, 1.0, 0),
        beta=LossSchedule(0.5, 1.0, 10),
        gamma=LossSchedule(0.2, 0.2, 0),
    )
    outputs = {
        "task_logits": torch.randn(4, 4, 8, 8),
        "representation": torch.randn(4, 128),
        "sparse_reconstruction": torch.randn(4, 128),
        "sparse_code": torch.randn(4, 64),
    }
    batch = {
        "target": torch.randint(0, 4, (4, 8, 8)),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }
    total, metrics = manager(outputs, batch, global_step=5)
    assert total.ndim == 0
    assert "loss_total" in metrics
