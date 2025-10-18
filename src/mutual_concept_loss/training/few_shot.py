"""Few-shot adaptation utilities for adapter-based fine-tuning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, MutableMapping

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..losses import LossManager
from ..models import LinearAdapter, SharedAutoencoderModel
from ..models.adapters import AdapterConfig
from .evaluation import evaluate_loop
from .loop import _group_ids_from_primitives, _precision_context, _prepare_batch


@dataclass
class FewShotConfig:
    """Configuration controlling adapter fine-tuning dynamics."""

    max_steps: int = 200
    eval_interval: int = 20
    log_interval: int = 10
    lr: float = 5e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp32"
    patience: int = 40
    metric: str = "task_iou"
    minimize_metric: bool = False


@dataclass
class FewShotResult:
    """Summary of the few-shot optimisation process."""

    support_history: list[Dict[str, float]] = field(default_factory=list)
    evaluation_history: list[Dict[str, float]] = field(default_factory=list)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    steps: int = 0
    wall_time: float = 0.0


class _MetricTracker:
    """Accumulates averages for scalar metrics."""

    def __init__(self) -> None:
        self._totals: MutableMapping[str, float] = {}
        self._count: int = 0

    def update(self, metrics: Dict[str, float]) -> None:
        self._count += 1
        for key, value in metrics.items():
            self._totals[key] = self._totals.get(key, 0.0) + float(value)

    def average(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {key: total / self._count for key, total in self._totals.items()}

    def reset(self) -> None:
        self._totals.clear()
        self._count = 0


def _cycle(loader: DataLoader[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches indefinitely by iterating over the loader repeatedly."""

    while True:
        for batch in loader:
            yield batch


def _ensure_adapter(model: SharedAutoencoderModel, adapter_config: AdapterConfig | None) -> None:
    """Attach a linear adapter when the model still uses the identity."""

    if isinstance(model.adapter, nn.Identity):
        hidden = model.config.bottleneck.hidden_dim
        config = adapter_config or AdapterConfig(
            input_dim=hidden,
            bottleneck_dim=max(4, hidden // 4),
        )
        model.attach_adapter(LinearAdapter(config))


def run_few_shot_adaptation(
    model: SharedAutoencoderModel,
    loss_manager: LossManager,
    support_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    query_loader: DataLoader[dict[str, torch.Tensor]] | None = None,
    config: FewShotConfig | None = None,
    adapter_config: AdapterConfig | None = None,
) -> FewShotResult:
    """Fine-tune linear adapters on support batches and evaluate on queries."""

    few_shot = config or FewShotConfig()
    device = torch.device(few_shot.device)
    model.to(device)
    _ensure_adapter(model, adapter_config)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.adapter_parameters():
        parameter.requires_grad = True

    optimizer = Adam(model.adapter_parameters(), lr=few_shot.lr, weight_decay=few_shot.weight_decay)
    context, scaler, _ = _precision_context(few_shot.precision, device)

    tracker = _MetricTracker()
    support_history: list[Dict[str, float]] = []
    evaluation_history: list[Dict[str, float]] = []
    best_metrics: Dict[str, float] = {}
    best_value: float | None = None
    patience_counter = 0
    step = 0
    start = time.perf_counter()
    iterator = _cycle(support_loader)
    model.train()
    try:
        while step < few_shot.max_steps:
            step += 1
            batch = next(iterator)
            prepared = _prepare_batch(batch, device)
            group_ids = (
                _group_ids_from_primitives(prepared["primitives"]) if "primitives" in prepared else None
            )
            optimizer.zero_grad(set_to_none=True)
            with context:
                outputs = model(prepared["input"])
                loss, metrics = loss_manager(
                    outputs,
                    prepared,
                    global_step=step,
                    group_ids=group_ids,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if few_shot.gradient_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), few_shot.gradient_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            metrics_map = {key: float(value) for key, value in metrics.items()}
            tracker.update(metrics_map)
            if step % few_shot.log_interval == 0 or step == 1:
                averaged = tracker.average()
                if averaged:
                    support_history.append(averaged)
                tracker.reset()

            should_eval = query_loader is not None and (
                step % few_shot.eval_interval == 0 or step == few_shot.max_steps
            )
            if should_eval:
                eval_metrics = evaluate_loop(
                    model,
                    loss_manager,
                    query_loader,
                    device=device,
                    step=step,
                )
                evaluation_history.append(eval_metrics)
                metric_value = eval_metrics.get(few_shot.metric)
                if metric_value is not None:
                    metric_float = float(metric_value)
                    if best_value is None:
                        best_value = metric_float
                        best_metrics = eval_metrics
                        patience_counter = 0
                    else:
                        improved = (
                            metric_float < best_value if few_shot.minimize_metric else metric_float > best_value
                        )
                        if improved:
                            best_value = metric_float
                            best_metrics = eval_metrics
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if few_shot.patience and patience_counter >= few_shot.patience:
                                break
    finally:
        wall_time = time.perf_counter() - start

    if not best_metrics and evaluation_history:
        best_metrics = evaluation_history[-1]

    return FewShotResult(
        support_history=support_history,
        evaluation_history=evaluation_history,
        best_metrics=best_metrics,
        steps=step,
        wall_time=wall_time,
    )
