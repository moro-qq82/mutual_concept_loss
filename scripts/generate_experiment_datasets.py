"""実験用の合成タスクデータセットを一括生成するスクリプト。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch

from mutual_concept_loss import GeneratorConfig, SyntheticTaskDataset


def _prepare_storage(
    num_samples: int,
    config: GeneratorConfig,
    num_primitives: int,
) -> Dict[str, torch.Tensor]:
    """生成した分割を格納するテンソル領域を確保する。"""

    size = config.grid_size
    channels = config.num_colors
    max_length = config.max_composition_length
    storage = {
        "inputs": torch.empty((num_samples, size, size, channels), dtype=torch.float32),
        "targets": torch.empty((num_samples, size, size, channels), dtype=torch.float32),
        "primitive_multi_hot": torch.empty((num_samples, num_primitives), dtype=torch.float32),
        "primitive_indices": torch.full((num_samples, max_length), -1, dtype=torch.int64),
        "sequence_length": torch.empty((num_samples,), dtype=torch.int64),
    }
    return storage


def _fill_storage(
    dataset: SyntheticTaskDataset,
    storage: Dict[str, torch.Tensor],
) -> None:
    """データセットからサンプルを取り出しバッファへ書き込む。"""

    for idx in range(len(dataset)):
        sample = dataset[idx]
        storage["inputs"][idx] = sample["input"]
        storage["targets"][idx] = sample["target"]
        storage["primitive_multi_hot"][idx] = sample["primitives"]
        storage["primitive_indices"][idx] = sample["primitive_indices"]
        storage["sequence_length"][idx] = sample["sequence_length"]


def _generate_split(
    name: str,
    num_samples: int,
    config: GeneratorConfig,
    seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    """分割データを生成してディスクへ保存する。"""

    dataset = SyntheticTaskDataset(num_samples=num_samples, config=config, seed=seed)
    storage = _prepare_storage(num_samples, config, len(dataset.generator.primitives))
    _fill_storage(dataset, storage)

    payload = {
        "data": storage,
        "generator_config": asdict(config),
        "seed": seed,
    }

    output_path = output_dir / f"{name}.pt"
    torch.save(payload, output_path)

    summary = {
        "name": name,
        "num_samples": num_samples,
        "seed": seed,
        "generator_config": asdict(config),
        "path": output_path.name,
    }
    return summary


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="生成データセットを書き出すディレクトリ")
    parser.add_argument("--train-samples", type=int, default=50000, help="学習用サンプル数")
    parser.add_argument("--val-samples", type=int, default=5000, help="検証用サンプル数")
    parser.add_argument("--test-samples", type=int, default=5000, help="テスト用サンプル数")
    parser.add_argument("--train-seed", type=int, default=1000, help="学習分割のシード")
    parser.add_argument("--val-seed", type=int, default=2000, help="検証分割のシード")
    parser.add_argument("--test-seed", type=int, default=3000, help="テスト分割のシード")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="グリッドの一辺の大きさ (Phase 1/2 の設定に合わせる)",
    )
    parser.add_argument(
        "--num-colors",
        type=int,
        default=4,
        help="使用する色チャネル数",
    )
    parser.add_argument(
        "--min-base-shapes",
        type=int,
        default=1,
        help="ベース図形の最小生成数",
    )
    parser.add_argument(
        "--max-base-shapes",
        type=int,
        default=3,
        help="ベース図形の最大生成数",
    )
    parser.add_argument(
        "--train-composition",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1, 2),
        help="学習分割の合成長 (min max)",
    )
    parser.add_argument(
        "--evaluation-composition",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(3, 3),
        help="検証・テスト分割の合成長 (min max)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON形式のメタデータを書き出すパス (省略時はoutput直下のmanifest.json)",
    )
    return parser.parse_args()


def _build_config(
    grid_size: int,
    num_colors: int,
    min_shapes: int,
    max_shapes: int,
    composition: Tuple[int, int],
) -> GeneratorConfig:
    """CLIパラメータからジェネレータ設定を構築する。"""

    min_comp, max_comp = composition
    return GeneratorConfig(
        grid_size=grid_size,
        num_colors=num_colors,
        min_composition_length=min_comp,
        max_composition_length=max_comp,
        min_base_shapes=min_shapes,
        max_base_shapes=max_shapes,
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _build_config(
        args.grid_size,
        args.num_colors,
        args.min_base_shapes,
        args.max_base_shapes,
        args.train_composition,
    )
    eval_config = _build_config(
        args.grid_size,
        args.num_colors,
        args.min_base_shapes,
        args.max_base_shapes,
        args.evaluation_composition,
    )

    splits = []
    splits.append(
        _generate_split(
            "train",
            args.train_samples,
            train_config,
            args.train_seed,
            output_dir,
        )
    )
    splits.append(
        _generate_split(
            "val",
            args.val_samples,
            eval_config,
            args.val_seed,
            output_dir,
        )
    )
    splits.append(
        _generate_split(
            "test",
            args.test_samples,
            eval_config,
            args.test_seed,
            output_dir,
        )
    )

    manifest_path = args.manifest or (output_dir / "manifest.json")
    manifest = {
        "description": "共有サブスペースX実験向けの合成タスク分割",
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
