"""Tests for the few-shot adaptation utilities."""

from __future__ import annotations

from mutual_concept_loss import GeneratorConfig
from mutual_concept_loss.losses import LossManager, SparseAutoencoderLoss, TaskLoss
from mutual_concept_loss.losses.share import SharedSubspaceLoss
from mutual_concept_loss.models import SharedAutoencoderModel
from mutual_concept_loss.training import (
    FewShotConfig,
    FewShotDataConfig,
    build_few_shot_loaders,
    run_few_shot_adaptation,
)


def _build_loss_manager() -> LossManager:
    task = TaskLoss()
    share = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, share, sparse)


def test_few_shot_adaptation_completes() -> None:
    """Adapter fine-tuning should run for a handful of steps on CPU."""

    model = SharedAutoencoderModel()
    loss_manager = _build_loss_manager()
    generator = GeneratorConfig(min_composition_length=3, max_composition_length=3)
    data_config = FewShotDataConfig(
        support_samples=8,
        query_samples=8,
        batch_size=2,
        generator=generator,
        seed=123,
    )
    support_loader, query_loader = build_few_shot_loaders(data_config)
    config = FewShotConfig(
        max_steps=4,
        eval_interval=2,
        log_interval=1,
        lr=1e-3,
        weight_decay=0.0,
        patience=3,
        device="cpu",
        precision="fp32",
    )
    result = run_few_shot_adaptation(
        model,
        loss_manager,
        support_loader,
        query_loader=query_loader,
        config=config,
    )
    assert result.steps <= config.max_steps
    assert result.support_history, "support metrics should be recorded"
    assert result.evaluation_history, "evaluation metrics should be recorded"
    assert result.best_metrics, "best metrics should be tracked"
