"""Few-shot fine-tuning entry point for adapter-based adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutual_concept_loss import (
    FewShotConfig,
    FewShotDataConfig,
    FewShotResult,
    GeneratorConfig,
    LossManager,
    SharedAutoencoderModel,
    SharedSubspaceLoss,
    SparseAutoencoderLoss,
    TaskLoss,
    build_few_shot_loaders,
    run_few_shot_adaptation,
)


def _build_loss_manager() -> LossManager:
    """Create the training loss manager used during fine-tuning."""

    task = TaskLoss()
    share = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, share, sparse)


def _load_checkpoint(model: SharedAutoencoderModel, path: Path, device: torch.device) -> None:
    """Load a checkpoint into the provided model."""

    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model")
    if state_dict is None:
        raise RuntimeError("checkpoint does not contain 'model' state")
    model.load_state_dict(state_dict)


def _serialize_result(result: FewShotResult) -> dict:
    """Convert the dataclass into a JSON-friendly dictionary."""

    return {
        "support_history": result.support_history,
        "evaluation_history": result.evaluation_history,
        "best_metrics": result.best_metrics,
        "steps": result.steps,
        "wall_time": result.wall_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the base model checkpoint")
    parser.add_argument("--support-samples", type=int, default=64, help="Number of support examples")
    parser.add_argument("--query-samples", type=int, default=256, help="Number of evaluation examples")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for both loaders")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum optimisation steps")
    parser.add_argument("--eval-interval", type=int, default=20, help="Steps between evaluations")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between support logs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adapter learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adapter weight decay")
    parser.add_argument("--patience", type=int, default=40, help="Number of evaluations without improvement")
    parser.add_argument("--device", type=str, default="cpu", help="Device for adaptation")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision setting (fp32/fp16/bf16)")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for dataset generation")
    parser.add_argument(
        "--min-composition",
        type=int,
        default=3,
        help="Minimum composition length for adaptation tasks",
    )
    parser.add_argument(
        "--max-composition",
        type=int,
        default=3,
        help="Maximum composition length for adaptation tasks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to store the adaptation summary",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SharedAutoencoderModel()
    _load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    generator = GeneratorConfig(
        min_composition_length=args.min_composition,
        max_composition_length=args.max_composition,
    )
    data_config = FewShotDataConfig(
        support_samples=args.support_samples,
        query_samples=args.query_samples,
        batch_size=args.batch_size,
        generator=generator,
        seed=args.seed,
    )
    support_loader, query_loader = build_few_shot_loaders(data_config)

    few_shot_config = FewShotConfig(
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        precision=args.precision,
    )
    loss_manager = _build_loss_manager()
    result = run_few_shot_adaptation(
        model,
        loss_manager,
        support_loader,
        query_loader=query_loader,
        config=few_shot_config,
    )

    payload = _serialize_result(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
