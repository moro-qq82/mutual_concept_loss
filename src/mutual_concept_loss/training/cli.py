"""共通の学習スクリプト向けユーティリティ。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from ..data import GeneratorConfig
from ..losses import LossManager, LossSchedule, SharedSubspaceLoss, SparseAutoencoderLoss, TaskLoss
from ..models import SharedAutoencoderModel
from ..utils.seed import set_seed
from .loop import CheckpointConfig, LoggingConfig, TrainingConfig, Trainer


@dataclass
class RunComponents:
    """学習ループ構築に必要な構成要素。"""

    model: SharedAutoencoderModel
    loss_manager: LossManager
    optimizer: torch.optim.Optimizer
    training: TrainingConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig


def add_common_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path | str = Path("outputs/training"),
    include_loss_weights: bool = False,
) -> None:
    """全ての学習スクリプトで共有する引数定義を追加する。"""

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_output_dir),
        help="成果物を保存するルートディレクトリ",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="任意のラン名を明示的に指定",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="計算に使用するデバイス ('cpu' か 'cuda')",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4096,
        help="学習用に生成するサンプル数",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=1024,
        help="検証用に生成するサンプル数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="学習時のバッチサイズ",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="実行する総ステップ数",
    )
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=1,
        help="勾配蓄積のステップ数",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="前向き計算で用いる数値精度",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="データローダ用ワーカープロセス数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="再現性のための乱数シード",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adamの学習率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decayの係数",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="TensorBoardログを無効化",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="CSVログを書き出さない",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="検証を走らせるステップ間隔",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="訓練指標を表示する間隔",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="検証時に処理する最大バッチ数 (省略時は全て)",
    )
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        help="最良チェックポイントにオプティマイザも保存",
    )
    parser.add_argument(
        "--min-composition",
        type=int,
        default=1,
        help="生成するタスクの最小合成長",
    )
    parser.add_argument(
        "--max-composition",
        type=int,
        default=3,
        help="生成するタスクの最大合成長",
    )

    if include_loss_weights:
        parser.add_argument(
            "--task-weight",
            type=float,
            default=1.0,
            help="タスク損失に掛ける重み",
        )
        parser.add_argument(
            "--shared-weight",
            type=float,
            default=1.0,
            help="共有サブスペース正則化の重み",
        )
        parser.add_argument(
            "--sparse-weight",
            type=float,
            default=1.0,
            help="疎オートエンコーダ損失の重み",
        )


def _build_loss_manager(*, alpha: float, beta: float, gamma: float) -> LossManager:
    """LossManagerを固定重みで構築する。"""

    task = TaskLoss()
    shared = SharedSubspaceLoss()
    sparse = SparseAutoencoderLoss()
    alpha_schedule = LossSchedule(alpha, alpha, 0)
    beta_schedule = LossSchedule(beta, beta, 0)
    gamma_schedule = LossSchedule(gamma, gamma, 0)
    return LossManager(
        task,
        shared,
        sparse,
        alpha=alpha_schedule,
        beta=beta_schedule,
        gamma=gamma_schedule,
    )


def build_run_components(
    args: argparse.Namespace,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> RunComponents:
    """引数に基づいてTrainer構成を組み立てる。"""

    model = SharedAutoencoderModel()
    loss_manager = _build_loss_manager(alpha=alpha, beta=beta, gamma=gamma)
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
        device=torch.device(args.device),
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
    return RunComponents(
        model=model,
        loss_manager=loss_manager,
        optimizer=optimizer,
        training=training,
        logging=logging,
        checkpoint=checkpoint,
    )


def run_training(
    args: argparse.Namespace,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, Dict[str, float]]:
    """Trainerを実行して指標を辞書で返す。"""

    set_seed(args.seed)
    components = build_run_components(args, alpha=alpha, beta=beta, gamma=gamma)
    trainer = Trainer(
        components.model,
        components.loss_manager,
        components.optimizer,
        components.training,
        components.logging,
        components.checkpoint,
    )
    metrics = trainer.fit()
    return {phase: {key: float(value) for key, value in phase_metrics.items()} for phase, phase_metrics in metrics.items()}

