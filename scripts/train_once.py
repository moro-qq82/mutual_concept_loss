"""最小構成で学習ループを1回走らせるコマンドラインスクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutual_concept_loss import (
    CheckpointConfig,
    GeneratorConfig,
    LoggingConfig,
    LossManager,
    SharedAutoencoderModel,
    SharedSubspaceLoss,
    SparseAutoencoderLoss,
    TaskLoss,
    TrainingConfig,
    Trainer,
    set_seed,
)


def _build_loss_manager() -> LossManager:
    """Create the default loss manager used during training."""

    task = TaskLoss()
    shared = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, shared, sparse)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training"),
        help="Directory where run artifacts are stored",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit name for the run directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g. 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=2048,
        help="Number of synthetic samples generated for training",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=512,
        help="Number of synthetic samples generated for validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during training",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Total training steps to execute",
    )
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for simulating larger batches",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="Numerical precision for the forward pass",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for dataset generation and initialization",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied to the optimizer",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging even if the package is available",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV logging for the run",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Frequency (in steps) to run validation",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Frequency (in steps) to report training metrics",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Optional maximum number of validation batches to evaluate",
    )
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        help="Persist optimizer state in the best checkpoint",
    )
    parser.add_argument(
        "--min-composition",
        type=int,
        default=1,
        help="Minimum composition length for generated tasks",
    )
    parser.add_argument(
        "--max-composition",
        type=int,
        default=2,
        help="Maximum composition length for generated tasks",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model = SharedAutoencoderModel()
    loss_manager = _build_loss_manager()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    generator = GeneratorConfig(
        min_composition_length=args.min_composition,
        max_composition_length=args.max_composition,
    )
    training = TrainingConfig(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        accumulate_steps=args.accumulate_steps,
        precision=args.precision,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
        eval_batches=args.eval_batches,
        train_generator=generator,
        val_generator=generator,
    )
    logging = LoggingConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        enable_tensorboard=not args.no_tensorboard,
        enable_csv=not args.no_csv,
    )
    checkpoint = CheckpointConfig(save_optimizer=args.save_optimizer)

    trainer = Trainer(
        model,
        loss_manager,
        optimizer,
        training,
        logging,
        checkpoint,
    )
    metrics = trainer.fit()

    summary = {phase: {k: float(v) for k, v in phase_metrics.items()} for phase, phase_metrics in metrics.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
