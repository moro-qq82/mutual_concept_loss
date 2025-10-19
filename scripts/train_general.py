"""共有サブスペースと疎AEを含む一般学習スクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutual_concept_loss.training.cli import add_common_run_arguments, run_training


def parse_args() -> argparse.Namespace:
    """共通引数に損失重みの指定を加えて解釈する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_arguments(
        parser,
        default_output_dir=Path("outputs/general_training"),
        include_loss_weights=True,
    )
    return parser.parse_args()


def main() -> None:
    """一般設定で学習ループを実行する。"""

    args = parse_args()
    metrics = run_training(
        args,
        alpha=args.task_weight,
        beta=args.shared_weight,
        gamma=args.sparse_weight,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

