"""Evaluation helpers exposed for standalone validation scripts."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from ..losses import LossManager
from .loop import evaluate_loop as _evaluate_loop


def evaluate_loop(
    model: nn.Module,
    loss_manager: LossManager,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device | str,
    step: int,
) -> Dict[str, float]:
    """Proxy function that calls the core evaluation helper."""

    return _evaluate_loop(model, loss_manager, dataloader, device=device, step=step)
