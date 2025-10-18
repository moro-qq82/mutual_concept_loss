"""Learning rate scheduler utilities for training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.optim import Optimizer


@dataclass
class CosineWarmupSchedulerConfig:
    """Configuration for a cosine decay scheduler with linear warmup."""

    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")


def build_cosine_warmup_scheduler(
    optimizer: Optimizer, config: CosineWarmupSchedulerConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct a scheduler that linearly warms up then cosine decays."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / float(max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / float(max(1, config.max_steps - config.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = config.min_lr / max(optimizer.defaults.get("lr", 1e-8), 1e-8)
        return float(min_factor + (1.0 - min_factor) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
