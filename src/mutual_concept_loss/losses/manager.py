"""Helper for combining multiple loss components with schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .share import SharedSubspaceLoss
from .sparse_autoencoder import SparseAutoencoderLoss
from .task import TaskLoss


@dataclass
class LossSchedule:
    """Piecewise linear schedule for loss weights."""

    initial: float = 1.0
    target: float = 1.0
    warmup_steps: int = 0

    def value(self, step: int) -> float:
        """Return the weight value for a given global step."""

        if self.warmup_steps <= 0 or self.initial == self.target:
            return float(self.target)
        ratio = min(1.0, max(0.0, step / float(self.warmup_steps)))
        return float(self.initial + (self.target - self.initial) * ratio)


class LossManager(nn.Module):
    """Aggregates the task, shared subspace, and sparse AE losses."""

    def __init__(
        self,
        task_loss: TaskLoss,
        shared_loss: SharedSubspaceLoss,
        sparse_loss: SparseAutoencoderLoss,
        *,
        alpha: LossSchedule | None = None,
        beta: LossSchedule | None = None,
        gamma: LossSchedule | None = None,
    ) -> None:
        super().__init__()
        self.task_loss = task_loss
        self.shared_loss = shared_loss
        self.sparse_loss = sparse_loss
        self.alpha = alpha or LossSchedule(1.0, 1.0, 0)
        self.beta = beta or LossSchedule(1.0, 1.0, 0)
        self.gamma = gamma or LossSchedule(1.0, 1.0, 0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        global_step: int,
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted sum of available losses."""

        weights = {
            "alpha": self.alpha.value(global_step),
            "beta": self.beta.value(global_step),
            "gamma": self.gamma.value(global_step),
        }

        task_value, task_metrics = self.task_loss(outputs["task_logits"], batch["target"])
        shared_ids = group_ids
        if shared_ids is None:
            shared_ids = batch.get("group_ids")
        if shared_ids is None:
            shared_ids = torch.arange(
                outputs["representation"].size(0), device=outputs["representation"].device
            )
        shared_value, shared_metrics = self.shared_loss(outputs["representation"], shared_ids)
        sparse_value, sparse_metrics = self.sparse_loss(
            outputs["sparse_reconstruction"],
            outputs["representation"].detach(),
            outputs["sparse_code"],
        )

        total = (
            weights["alpha"] * task_value
            + weights["beta"] * shared_value
            + weights["gamma"] * sparse_value
        )
        metrics: dict[str, torch.Tensor] = {}
        metrics.update(task_metrics)
        metrics.update(shared_metrics)
        metrics.update(sparse_metrics)
        metrics.update({
            "loss_total": total.detach(),
            "loss_alpha": torch.tensor(weights["alpha"], device=total.device),
            "loss_beta": torch.tensor(weights["beta"], device=total.device),
            "loss_gamma": torch.tensor(weights["gamma"], device=total.device),
        })
        return total, metrics
