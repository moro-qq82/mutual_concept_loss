"""Tests for the Phase 3 training loop implementation."""

from __future__ import annotations

from pathlib import Path

import torch

from mutual_concept_loss.losses import (
    LossManager,
    SparseAutoencoderLoss,
    TaskLoss,
)
from mutual_concept_loss.losses.share import SharedSubspaceLoss
from mutual_concept_loss.models import SharedAutoencoderModel
from mutual_concept_loss.training import (
    CheckpointConfig,
    LoggingConfig,
    TrainingConfig,
    Trainer,
)


def _build_loss_manager() -> LossManager:
    task = TaskLoss()
    shared = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, shared, sparse)


def test_trainer_runs_single_epoch(tmp_path: Path) -> None:
    """Trainer should complete a short run and emit checkpoints."""

    model = SharedAutoencoderModel()
    loss_manager = _build_loss_manager()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training = TrainingConfig(
        train_samples=24,
        val_samples=8,
        batch_size=4,
        max_steps=4,
        accumulate_steps=1,
        precision="fp32",
        device="cpu",
        num_workers=0,
        eval_batches=1,
    )
    logging = LoggingConfig(
        output_dir=tmp_path,
        run_name="test_run",
        log_interval=1,
        eval_interval=2,
        enable_tensorboard=False,
        enable_csv=False,
        save_config=False,
    )
    checkpoint = CheckpointConfig(metric="task_iou", mode="max", save_optimizer=False)
    trainer = Trainer(
        model,
        loss_manager,
        optimizer,
        training,
        logging,
        checkpoint,
    )
    metrics = trainer.fit()
    assert "train" in metrics
    assert "val" in metrics
    run_dir = tmp_path / "runs" / "test_run"
    assert run_dir.exists()
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    assert checkpoint_path.exists()
