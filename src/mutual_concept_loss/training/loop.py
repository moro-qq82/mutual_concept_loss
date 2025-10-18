"""Training loop implementation with logging and checkpoints."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, MutableMapping, Optional

import torch
from torch import nn
from torch.cuda import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
try:  # pragma: no cover - import guard
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc]

from ..data.generator import GeneratorConfig, SyntheticTaskDataset
from ..data.sampler import GroupedBatchSampler
from ..losses import LossManager
from ..models import SharedAutoencoderModel


@dataclass
class OptimizerConfig:
    """Hyperparameters for the Adam optimizer used in training."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class LoggingConfig:
    """Configuration for CSV and TensorBoard logging."""

    output_dir: Path = Path("outputs")
    run_name: str | None = None
    log_interval: int = 10
    eval_interval: int = 100
    enable_tensorboard: bool = True
    enable_csv: bool = True
    flush_interval: int = 50
    save_config: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint behaviour controlled by validation metric comparisons."""

    metric: str = "task_iou"
    mode: str = "max"
    save_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")


@dataclass
class TrainingConfig:
    """High level hyperparameters for the training loop."""

    train_samples: int = 1024
    val_samples: int = 256
    batch_size: int = 32
    max_steps: int = 1000
    accumulate_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    precision: str = "fp32"
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    seed: int = 0
    eval_batches: int | None = None
    train_generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    val_generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be positive")
        if self.train_samples <= 0 or self.val_samples <= 0:
            raise ValueError("dataset sizes must be positive")
        if self.eval_batches is not None and self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive when provided")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of 'fp32', 'fp16', or 'bf16'")


class _MetricTracker:
    """Utility class to maintain running averages for scalar metrics."""

    def __init__(self) -> None:
        self._totals: MutableMapping[str, float] = {}
        self._count: int = 0

    def update(self, metrics: Dict[str, float]) -> None:
        self._count += 1
        for key, value in metrics.items():
            total = self._totals.get(key, 0.0)
            self._totals[key] = total + float(value)

    def average(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {key: total / self._count for key, total in self._totals.items()}

    def reset(self) -> None:
        self._totals.clear()
        self._count = 0


class _CSVLogger:
    """Minimal CSV logger that appends metrics with a consistent header."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._header: list[str] | None = None

    def log(self, step: int, phase: str, metrics: Dict[str, float]) -> None:
        row = {"step": float(step), "phase": phase}
        row.update(metrics)
        if self._header is None:
            self._header = list(row.keys())
            self._writer.writerow(self._header)
        ordered = [row.get(column, math.nan) for column in self._header]
        self._writer.writerow(ordered)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class _CheckpointManager:
    """Tracks the best validation metric and persists checkpoints."""

    def __init__(self, directory: Path, config: CheckpointConfig) -> None:
        self.directory = directory
        self.config = config
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_value: float | None = None
        self.best_path = self.directory / "best.pt"

    def update(
        self,
        *,
        step: int,
        metrics: Dict[str, float],
        model: nn.Module,
        optimizer: Optimizer | None,
    ) -> None:
        metric_value = metrics.get(self.config.metric)
        if metric_value is None:
            return
        metric_value = float(metric_value)
        if self.best_value is None or self._is_better(metric_value, self.best_value):
            self.best_value = metric_value
            payload = {"model": model.state_dict(), "step": step, self.config.metric: metric_value}
            if self.config.save_optimizer and optimizer is not None:
                payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, self.best_path)

    def _is_better(self, current: float, best: float) -> bool:
        if self.config.mode == "max":
            return current >= best
        return current <= best


def _precision_context(precision: str, device: torch.device) -> tuple[amp.autocast, Optional[amp.GradScaler], torch.dtype | None]:
    """Return autocast context, scaler, and dtype based on precision setting."""

    if precision == "fp32":
        return amp.autocast(enabled=False), None, None
    if precision == "fp16":
        scaler = amp.GradScaler(enabled=device.type == "cuda")
        return amp.autocast(device_type=device.type, dtype=torch.float16), scaler, torch.float16
    if precision == "bf16":
        return amp.autocast(device_type=device.type, dtype=torch.bfloat16), None, torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def _write_config_yaml(path: Path, config: TrainingConfig, logging: LoggingConfig, optimizer: OptimizerConfig, checkpoint: CheckpointConfig) -> None:
    """Serialize configurations as a simple YAML document."""

    def _format_value(value: object, indent: int = 0) -> str:
        prefix = " " * indent
        if isinstance(value, dict):
            if not value:
                return f"{prefix}{{}}"
            lines = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(_format_value(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {item}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return f"{prefix}[]"
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(_format_value(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {item}")
            return "\n".join(lines)
        return f"{prefix}{value}"

    merged = {
        "training": asdict(config),
        "logging": asdict(logging),
        "optimizer": asdict(optimizer),
        "checkpoint": asdict(checkpoint),
    }
    text_lines = ["# Auto-generated configuration", _format_value(merged)]
    path.write_text("\n".join(text_lines))


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensors to the target device for computation."""

    prepared: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
    return prepared


def _group_ids_from_primitives(primitives: torch.Tensor) -> torch.Tensor:
    """Generate group identifiers from multi-hot primitive indicators."""

    if primitives.dim() != 2:
        raise ValueError("primitives must be shaped as (B, P)")
    weights = 2 ** torch.arange(primitives.size(1), device=primitives.device, dtype=torch.float32)
    encoded = (primitives > 0.5).float() * weights
    return encoded.sum(dim=1).long()


class Trainer:
    """Orchestrates dataset loading, optimization, logging, and checkpoints."""

    def __init__(
        self,
        model: SharedAutoencoderModel,
        loss_manager: LossManager,
        optimizer: Optimizer,
        training: TrainingConfig,
        logging: LoggingConfig,
        checkpoint: CheckpointConfig,
        *,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.training = training
        self.logging = logging
        self.checkpoint = checkpoint
        self.scheduler = scheduler
        self.device = torch.device(training.device)
        self.model.to(self.device)

        self._run_dir = self._create_run_directory()
        if self.logging.save_config:
            _write_config_yaml(
                self._run_dir / "config.yaml",
                training,
                logging,
                OptimizerConfig(
                    lr=optimizer.defaults.get("lr", 0.0),
                    weight_decay=optimizer.defaults.get("weight_decay", 0.0),
                    betas=optimizer.defaults.get("betas", (0.0, 0.0)),
                ),
                checkpoint,
            )
        if logging.enable_tensorboard and SummaryWriter is None:
            raise RuntimeError("TensorBoard support requires torch.utils.tensorboard to be installed")
        self._tensorboard = self._create_writer() if logging.enable_tensorboard else None
        self._csv = _CSVLogger(self._run_dir / "metrics.csv") if logging.enable_csv else None
        self._checkpoint = _CheckpointManager(self._run_dir / "checkpoints", checkpoint)
        self._train_loader = self._create_dataloader(
            training.train_samples,
            training.train_generator,
            grouped=True,
            seed_offset=0,
        )
        self._val_loader = self._create_dataloader(
            training.val_samples,
            training.val_generator,
            grouped=False,
            seed_offset=1_000_000,
        )
        self._train_iter: Iterator[dict[str, torch.Tensor]] | None = None

    def fit(self) -> Dict[str, Dict[str, float]]:
        """Execute the training loop and return final metrics."""

        context, scaler, _ = _precision_context(self.training.precision, self.device)
        train_tracker = _MetricTracker()
        result: Dict[str, Dict[str, float]] = {}
        step_start = time.perf_counter()
        try:
            for step in range(1, self.training.max_steps + 1):
                batch = self._next_batch()
                prepared = _prepare_batch(batch, self.device)
                group_ids = _group_ids_from_primitives(prepared["primitives"]) if "primitives" in prepared else None
                with context:
                    outputs = self.model(prepared["input"])
                    loss, metrics = self.loss_manager(
                        outputs,
                        prepared,
                        global_step=step,
                        group_ids=group_ids,
                    )
                    loss = loss / float(self.training.accumulate_steps)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                should_step = step % self.training.accumulate_steps == 0 or step == self.training.max_steps
                if should_step:
                    if self.training.gradient_clip_norm is not None:
                        if scaler is not None:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.gradient_clip_norm)
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                metrics = {key: float(value) for key, value in metrics.items()}
                metrics["step_time"] = time.perf_counter() - step_start
                train_tracker.update(metrics)
                if step % self.logging.log_interval == 0 or step == 1:
                    averaged = train_tracker.average()
                    self._log_metrics("train", step, averaged)
                    result["train"] = averaged
                    train_tracker.reset()
                if step % self.logging.eval_interval == 0 or step == self.training.max_steps:
                    eval_metrics = self.evaluate(step)
                    self._log_metrics("val", step, eval_metrics)
                    self._checkpoint.update(step=step, metrics=eval_metrics, model=self.model, optimizer=self.optimizer)
                    result["val"] = eval_metrics
                step_start = time.perf_counter()
        finally:
            if self._tensorboard is not None:
                self._tensorboard.close()
            if self._csv is not None:
                self._csv.close()
        return result

    def evaluate(self, step: int) -> Dict[str, float]:
        """Run the validation loop for a fixed number of batches."""

        tracker = _MetricTracker()
        self.model.eval()
        batches = 0
        with torch.no_grad():
            for batch in self._val_loader:
                prepared = _prepare_batch(batch, self.device)
                group_ids = _group_ids_from_primitives(prepared["primitives"]) if "primitives" in prepared else None
                outputs = self.model(prepared["input"])
                _, metrics = self.loss_manager(
                    outputs,
                    prepared,
                    global_step=step,
                    group_ids=group_ids,
                )
                tracker.update({key: float(value) for key, value in metrics.items()})
                batches += 1
                if self.training.eval_batches is not None and batches >= self.training.eval_batches:
                    break
        self.model.train()
        averaged = tracker.average()
        if not averaged:
            return {}
        return averaged

    def _next_batch(self) -> dict[str, torch.Tensor]:
        if self._train_iter is None:
            self._train_iter = iter(self._train_loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self._train_loader)
            batch = next(self._train_iter)
        return batch

    def _log_metrics(self, phase: str, step: int, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        if self._tensorboard is not None:
            for key, value in metrics.items():
                self._tensorboard.add_scalar(f"{phase}/{key}", value, step)
                if step % self.logging.flush_interval == 0:
                    self._tensorboard.flush()
        if self._csv is not None:
            self._csv.log(step, phase, metrics)

    def _create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = self.logging.run_name or timestamp
        run_dir = self.logging.output_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _create_writer(self) -> SummaryWriter | None:
        log_dir = self._run_dir / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is None:
            return None
        return SummaryWriter(log_dir=str(log_dir))

    def _create_dataloader(
        self,
        num_samples: int,
        generator_config: GeneratorConfig,
        *,
        grouped: bool,
        seed_offset: int,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        dataset = SyntheticTaskDataset(
            num_samples=num_samples,
            config=generator_config,
            seed=self.training.seed + seed_offset,
        )
        if grouped:
            generator = torch.Generator()
            generator.manual_seed(self.training.seed + seed_offset)
            sampler = GroupedBatchSampler(
                dataset,
                self.training.batch_size,
                generator=generator,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.training.num_workers,
            )
        return DataLoader(
            dataset,
            batch_size=self.training.batch_size,
            shuffle=False,
            num_workers=self.training.num_workers,
        )


def evaluate_loop(
    model: nn.Module,
    loss_manager: LossManager,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device | str,
    step: int,
) -> Dict[str, float]:
    """Utility evaluation loop for standalone validation."""

    tracker = _MetricTracker()
    model_device = torch.device(device)
    model.to(model_device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prepared = _prepare_batch(batch, model_device)
            group_ids = _group_ids_from_primitives(prepared["primitives"]) if "primitives" in prepared else None
            outputs = model(prepared["input"])
            _, metrics = loss_manager(outputs, prepared, global_step=step, group_ids=group_ids)
            tracker.update({key: float(value) for key, value in metrics.items()})
    model.train()
    return tracker.average()
