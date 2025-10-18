"""Evaluation helpers exposed for standalone validation scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..losses import LossManager
from ..data.generator import GeneratorConfig, SyntheticTaskDataset
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


@dataclass
class EvaluationDataConfig:
    """Configuration for zero-shot evaluation datasets."""

    num_samples: int = 256
    batch_size: int = 32
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 0
    num_workers: int = 0


@dataclass
class FewShotDataConfig:
    """Dataset sizes for adapter-based fine-tuning."""

    support_samples: int = 32
    query_samples: int = 128
    batch_size: int = 16
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 0
    num_workers: int = 0


def build_zero_shot_dataloader(config: EvaluationDataConfig) -> DataLoader[dict[str, torch.Tensor]]:
    """Construct a dataloader for zero-shot evaluation."""

    dataset = SyntheticTaskDataset(
        num_samples=config.num_samples,
        config=config.generator,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def build_few_shot_loaders(
    config: FewShotDataConfig,
) -> Tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Create support and query loaders for few-shot adaptation."""

    support_dataset = SyntheticTaskDataset(
        num_samples=config.support_samples,
        config=config.generator,
        seed=config.seed,
    )
    query_dataset = SyntheticTaskDataset(
        num_samples=config.query_samples,
        config=config.generator,
        seed=config.seed + 10_000,
    )
    support_loader = DataLoader(
        support_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return support_loader, query_loader
