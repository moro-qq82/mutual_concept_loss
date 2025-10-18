"""Run Phase 5 representation analysis (CKA, Grassmann, sparse stats)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
from mutual_concept_loss.analysis import (
    collect_representations,
    group_by_label,
    pairwise_grassmann_distances,
    pairwise_linear_cka,
    summarize_sparse_codes,
)

try:  # pragma: no cover - optional dependency for visualization
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _build_loss_manager() -> LossManager:
    """Instantiate the default loss manager for metric reuse."""

    task = TaskLoss()
    share = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    return LossManager(task, share, sparse)


def _load_checkpoint(model: SharedAutoencoderModel, path: Path, device: torch.device) -> int:
    """Restore model parameters from a checkpoint."""

    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model")
    if state_dict is None:
        raise RuntimeError("checkpoint does not contain 'model' state")
    model.load_state_dict(state_dict)
    return int(payload.get("step", 0))


def _save_heatmap(matrix: torch.Tensor, labels: Sequence[str], title: str, path: Path) -> None:
    """Persist a matrix visualization if matplotlib is available."""

    if plt is None:
        raise RuntimeError("matplotlib is required to save heatmaps")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix.numpy(), interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the trained model checkpoint")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples for analysis")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for analysis dataloader")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dataset generation")
    parser.add_argument(
        "--min-composition",
        type=int,
        default=1,
        help="Minimum composition length for analysis tasks",
    )
    parser.add_argument(
        "--max-composition",
        type=int,
        default=3,
        help="Maximum composition length for analysis tasks",
    )
    parser.add_argument(
        "--feature-keys",
        nargs="+",
        default=("representation", "sparse_code", "primitive_logits"),
        help="Model output keys included in the CKA study",
    )
    parser.add_argument(
        "--grassmann-feature",
        type=str,
        default="representation",
        help="Model output key used for Grassmann analysis",
    )
    parser.add_argument(
        "--grassmann-k",
        type=int,
        default=8,
        help="Number of principal components for Grassmann distance",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store numeric summaries",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=Path,
        default=None,
        help="Optional directory to write heatmap images",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also compute evaluation metrics on the analysis split",
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

    collection = collect_representations(
        model,
        dataloader,
        device=device,
        keys=tuple(args.feature_keys),
    )

    feature_names, cka_matrix = pairwise_linear_cka(
        collection.features.values(), labels=list(collection.features.keys())
    )
    if args.grassmann_feature not in collection.features:
        raise KeyError(
            f"Feature '{args.grassmann_feature}' not collected; update --feature-keys accordingly"
        )
    grassmann_labels, grassmann_matrix = pairwise_grassmann_distances(
        group_by_label(collection.features[args.grassmann_feature], collection.sequence_length),
        k=args.grassmann_k,
    )
    if "sparse_code" not in collection.features:
        raise KeyError("'sparse_code' feature required for sparse analysis; add to --feature-keys")
    sparse_summary = summarize_sparse_codes(collection.features["sparse_code"], collection.primitives)

    result = {
        "step": step,
        "cka": {
            "features": list(feature_names),
            "matrix": cka_matrix.tolist(),
        },
        "grassmann": {
            "groups": list(grassmann_labels),
            "matrix": grassmann_matrix.tolist(),
            "k": args.grassmann_k,
        },
        "sparse": sparse_summary.to_dict(),
    }

    if args.evaluate:
        loss_manager = _build_loss_manager()
        metrics = evaluate_loop(model, loss_manager, dataloader, device=device, step=step)
        result["evaluation"] = metrics

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))

    if args.heatmap_dir is not None:
        if plt is None:
            raise RuntimeError("matplotlib is required to generate heatmaps")
        _save_heatmap(cka_matrix, feature_names, "CKA", args.heatmap_dir / "cka.png")
        _save_heatmap(grassmann_matrix, grassmann_labels, "Grassmann Distance", args.heatmap_dir / "grassmann.png")


if __name__ == "__main__":
    main()
