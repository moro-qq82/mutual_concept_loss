"""Command line utility for zero-shot evaluation on unseen compositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutual_concept_loss import (
    EvaluationDataConfig,
    GeneratorConfig,
    LossManager,
    SharedAutoencoderModel,
    SparseAutoencoderLoss,
    TaskLoss,
    SharedSubspaceLoss,
    build_zero_shot_dataloader,
    evaluate_loop,
)


def _build_loss_manager() -> LossManager:
    """Instantiate the default loss manager used during training."""

    task = TaskLoss()
    share = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, share, sparse)


def _load_checkpoint(model: SharedAutoencoderModel, path: Path, device: torch.device) -> int:
    """Restore model weights from disk and return the stored step."""

    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model")
    if state_dict is None:
        raise RuntimeError("checkpoint does not contain 'model' state")
    model.load_state_dict(state_dict)
    return int(payload.get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the trained model checkpoint")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of evaluation samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Target device for inference")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dataset generation")
    parser.add_argument(
        "--min-composition",
        type=int,
        default=3,
        help="Minimum composition length for evaluation tasks",
    )
    parser.add_argument(
        "--max-composition",
        type=int,
        default=3,
        help="Maximum composition length for evaluation tasks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store the resulting metrics as JSON",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SharedAutoencoderModel()
    step = _load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    generator = GeneratorConfig(
        min_composition_length=args.min_composition,
        max_composition_length=args.max_composition,
    )
    data_config = EvaluationDataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        generator=generator,
        seed=args.seed,
    )
    dataloader = build_zero_shot_dataloader(data_config)
    loss_manager = _build_loss_manager()
    metrics = evaluate_loop(model, loss_manager, dataloader, device=device, step=step)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
