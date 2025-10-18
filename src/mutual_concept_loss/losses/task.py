"""Primary task loss for grid prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TaskLossConfig:
    """Configuration for the task loss."""

    mode: str = "cross_entropy"
    reduction: str = "mean"


class TaskLoss(nn.Module):
    """Computes either cross-entropy or (1 - IoU) for grid prediction."""

    def __init__(self, config: TaskLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or TaskLossConfig()
        if self.config.mode not in {"cross_entropy", "one_minus_iou"}:
            raise ValueError("mode must be 'cross_entropy' or 'one_minus_iou'")
        if self.config.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the supervised loss and auxiliary metrics."""

        if logits.dim() != 4:
            raise ValueError("logits must be shaped as (B, C, H, W)")
        num_classes = logits.size(1)
        target_indices = self._prepare_targets(targets, num_classes)

        with torch.no_grad():
            iou = self._compute_mean_iou(logits, target_indices, num_classes)

        if self.config.mode == "cross_entropy":
            loss = F.cross_entropy(logits, target_indices, reduction=self.config.reduction)
        else:
            loss = 1.0 - iou
            if self.config.reduction == "sum":
                loss = loss * logits.size(0)

        metrics = {
            "task_iou": iou.detach(),
            "task_loss": loss.detach(),
        }
        return loss, metrics

    def _prepare_targets(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert dense or one-hot targets to integer indices."""

        if targets.dim() == 3:
            return targets.long()
        if targets.dim() == 4:
            if targets.size(-1) != num_classes:
                raise ValueError("one-hot targets must have the same channel count as logits")
            return targets.argmax(dim=-1)
        raise ValueError("targets must be shaped as (B, H, W) or (B, H, W, C)")

    def _compute_mean_iou(
        self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Compute mean intersection-over-union for segmentation logits."""

        predictions = logits.argmax(dim=1)
        pred_one_hot = F.one_hot(predictions, num_classes=num_classes).permute(0, 3, 1, 2)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        pred_one_hot = pred_one_hot.float()
        target_one_hot = target_one_hot.float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(1, 2, 3))
        union = pred_one_hot.sum(dim=(1, 2, 3)) + target_one_hot.sum(dim=(1, 2, 3)) - intersection
        iou = intersection / (union + 1e-6)
        return iou.mean()
